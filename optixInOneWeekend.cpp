//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// gl_interopの前にincludeされる必要がある
#include <glad/glad.h> 

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

// sampleConfig.h.in から自動生成されるヘッダーファイルをinclude
// ディレクトリへのパスなどの環境変数が定義されている
#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixInOneWeekend.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool resize_dirty   = false;
bool minimized      = false;

// Camera state 
bool camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 16;

template <typename T>
struct Record 
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord   = Record<RayGenData>;
using MissRecord     = Record<MissData>;
using HitGroupRecord = Record<HitGroupRecord>;

struct CallableProgram
{
    OptixProgramGroup prg = nullptr;
    uint32_t          id  = 0;
};

struct GeometryAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    uint32_t num_sbt_records;
};

struct OneWeekendState
{
    OptixDeviceContext context = 0;

    // Instance Acceleration Structure の Traversable handle
    OptixTraversableHandle      ias_handle               = 0;
    CUdeviceptr                 d_ias_output_buffer      = 0;
    void*                       d_sphere_data            = nullptr;
    void*                       d_mesh_data              = nullptr;

    OptixModule                 module                   = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline               pipeline                 = nullptr;

    OptixProgramGroup           raygen_prg               = nullptr;
    OptixProgramGroup           miss_prg                 = nullptr;
    OptixProgramGroup           sphere_hit_prg           = nullptr;
    OptixProgramGroup           triangle_hit_prg         = nullptr;

    CUstream                    stream                   = 0;
    Params                      params;
    Params*                     d_params;

    OptixShaderBindingTable     sbt                      = {};
};

// GLFW callbacks ------------------------------------------------
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos; 

    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button; 
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else 
    {
        mouse_button = -1;
    }
}

// -----------------------------------------------------------------------
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    // 左クリック中にマウスが移動した場合は、注視点を固定してカメラを動かす
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params.width, params.height);
        camera_changed = true;
    }
    // 右クリック中にマウスが移動した場合は、カメラの原点を固定して注視点を動かす
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params.height);
        camera_changed = true;
    }
}

// -----------------------------------------------------------------------
static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // ウィンドウが最小化された時に、最小化される前のウィンドウの解像度を保存しておく
    if( minimized )
        return;

    // ウィンドウサイズが最小でも 1 x 1 になるようにする
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}

// -----------------------------------------------------------------------
static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

// -----------------------------------------------------------------------
static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        // Q or Esc -> 終了
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}

// -----------------------------------------------------------------------
static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}

// -----------------------------------------------------------------------
OptixAabb sphereBound(const SphereData& sphere)
{
    const float3 center = sphere.center;
    const float radius = sphere.radius;
    // SphereのAxis-aligned bounding box を返す
    return OptixAabb {
        center.x - radius, center.y - radius, center.z - radius, 
        center.x + radius, center.y + radius, center.z + radius
    };
}

// -----------------------------------------------------------------------
void initLaunchParams( OneWeekendState& state )
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer), 
        state.params.width * state.params.height * sizeof(float4)
    ));
    state.params.frame_buffer = nullptr;

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    state.params.handle = state.ias_handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

// -----------------------------------------------------------------------
void handleCameraUpdate( Params& params )
{
    if (!camera_changed)
        return;

    camera_changed = false;
    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

// -----------------------------------------------------------------------
void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                params.width * params.height * sizeof( float4 )
                ) );
}

// -----------------------------------------------------------------------
void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}

// -----------------------------------------------------------------------
void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, OneWeekendState& state )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( state.d_params ),
                &state.params, sizeof( Params ),
                cudaMemcpyHostToDevice, state.stream
                ) );

    OPTIX_CHECK( optixLaunch(
                state.pipeline,
                state.stream,
                reinterpret_cast<CUdeviceptr>( state.d_params ),
                sizeof( Params ),
                &state.sbt,
                state.params.width,   // launch width
                state.params.height,  // launch height
                1                     // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

// -----------------------------------------------------------------------
void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}

// -----------------------------------------------------------------------
static void contextLogCallback(uint32_t level, const char* tag, const char* msg, void* /* callback_data */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << msg << "\n";
}

// -----------------------------------------------------------------------
void createContext( OneWeekendState& state )
{
    // CUDAの初期化
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext   cu_ctx = 0;
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction  = &contextLogCallback;
    options.logCallbackLevel     = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

// -----------------------------------------------------------------------
uint32_t getNumSbtRecords(const std::vector<uint32_t>& sbt_indices)
{
    std::vector<uint32_t> sbt_counter;
    for (const uint32_t& sbt_idx : sbt_indices)
    {
        auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
        if (sbt_counter.empty() || itr == sbt_counter.end())
            sbt_counter.emplace_back(sbt_idx);
    }
    return static_cast<uint32_t>(sbt_counter.size());
}

// -----------------------------------------------------------------------
void buildGAS( OneWeekendState& state, GeometryAccelData& gas, OptixBuildInput& build_input)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context, 
        &accel_options, 
        &build_input, 
        1, 
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), 
        compacted_size_offset + 8
    ));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset );

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &build_input, 
        1, 
        d_temp_buffer, 
        gas_buffer_sizes.tempSizeInBytes, 
        d_buffer_temp_output_gas_and_compacted_size, 
        gas_buffer_sizes.outputSizeInBytes, 
        &gas.handle, 
        &emit_property, 
        1
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    CUdeviceptr d_gas_output_buffer = 0;
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.d_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas.handle, gas.d_output_buffer, compacted_gas_size, &gas.handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// -----------------------------------------------------------------------
void buildMeshGAS(
    OneWeekendState& state, 
    GeometryAccelData& gas,
    const std::vector<float3>& vertices, 
    const std::vector<int3>& indices, 
    const std::vector<uint32_t>& sbt_indices
)
{
    CUdeviceptr d_vertices = 0;
    const size_t vertices_size = vertices.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices), 
        vertices.data(), vertices_size, 
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_indices = 0;
    const size_t indices_size = indices.size() * sizeof(int3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(), indices_size, 
        cudaMemcpyHostToDevice 
    ));

    CUdeviceptr d_sbt_indices = 0;
    const size_t sbt_indices_size = sbt_indices.size() * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sbt_indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void**>(d_sbt_indices),
        sbt_indices.data(), sbt_indices_size,
        cudaMemcpyHostToDevice
    ));

    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);

    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput mesh_input = {};
    mesh_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    mesh_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    mesh_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    mesh_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    mesh_input.triangleArray.vertexBuffers = &d_vertices;
    mesh_input.triangleArray.flags = input_flags;
    mesh_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    mesh_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    mesh_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
    mesh_input.triangleArray.numSbtRecords = num_sbt_records;
    mesh_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
    mesh_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    mesh_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(state, gas, mesh_input);
}

// -----------------------------------------------------------------------
void buildSphereGAS(
    OneWeekendState& state, 
    GeometryAccelData& gas,
    const std::vector<SphereData>& spheres, 
    const std::vector<uint32_t>& sbt_indices
)
{
    std::vector<OptixAabb> aabb;
    std::transform(spheres.begin(), spheres.end(), std::back_inserter(aabb),
        [](const SphereData& sphere) { return sphereBound(sphere); });

    CUdeviceptr d_aabb_buffer;
    const size_t aabb_size = sizeof(OptixAabb) * aabb.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer),
        aabb.data(), aabb_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_sbt_indices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sizeof(uint32_t) * sbt_indices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_indices),
        sbt_indices.data(), sizeof(uint32_t) * sbt_indices.size(),
        cudaMemcpyHostToDevice
    ));

    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);

    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    sphere_input.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(spheres.size());
    sphere_input.customPrimitiveArray.flags = input_flags;
    sphere_input.customPrimitiveArray.numSbtRecords = num_sbt_records;
    sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
    sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    sphere_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(state, gas, sphere_input);
}

