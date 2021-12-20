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

// OptiX SDK 提供のヘッダーファイルのinclude
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
#include "random.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool resize_dirty   = false;
bool minimized      = false;

// カメラ
bool camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// マウス
int32_t mouse_button = -1;

// 1度のカーネル起動におけるピクセルあたりのサンプル数
int32_t samples_per_launch = 16;

// Shader binding tableを構成するシェーダーレコードでヘッダーと任意のデータからなる。
// ヘッダーサイズはOptiX 7.4ではOPTIX_SBT_RECORD_HEADER_SIZE (32 bytes)で固定の値である。
// データはユーザー定義のデータ型を格納することが可能。ただし、Shader binding table内で
// 複数のレコードを保持できるHitGroup record, Miss record, Callables recordはそれぞれで
// レコードサイズが等しい必要がある。
template <typename T>
struct Record 
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord   = Record<RayGenData>;
using MissRecord     = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;
using EmptyRecord    = Record<EmptyData>;

// Direct/Continuation callable プログラムをデバイス(GPU)側で呼ぶには、
// OptixDeviceContextが作られてから、Callableプログラムが作られた順番(ID)
// が必要なので、プログラム作成時にIDを紐づけておく
struct CallableProgram
{
    OptixProgramGroup program = nullptr;
    uint32_t          id      = 0;
};

// Geometry acceleration structure (GAS) 用
// GASのtraversable handleをOptixInstanceに紐づける際に、
// GASが保持するSBT recordの数がわかると、
// Instanceのsbt offsetを一括で構築しやすい
struct GeometryAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    uint32_t num_sbt_records;
};

// Instance acceleration structure (IAS) 用
// 
struct InstanceAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    CUdeviceptr d_instances_buffer;
};

enum class ShapeType
{
    Mesh, 
    Sphere
};

struct OneWeekendState
{
    OptixDeviceContext context = 0;

    // シーン全体のInstance acceleration structure
    InstanceAccelData           ias                      = {};
    // GPU上におけるシーンの球体データ全てを格納している配列のポインタ
    void*                       d_sphere_data            = nullptr;
    // GPU上におけるシーンの三角形データ全てを格納している配列のポインタ
    void*                       d_mesh_data              = nullptr;

    OptixModule                 module                   = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline               pipeline                 = nullptr;

    // Ray generation プログラム 
    OptixProgramGroup           raygen_prg               = nullptr;
    // Miss プログラム
    OptixProgramGroup           miss_prg                 = nullptr;

    // 球体用のHitGroup プログラム
    OptixProgramGroup           sphere_hitgroup_prg      = nullptr;
    // メッシュ用のHitGroupプログラム
    OptixProgramGroup           mesh_hitgroup_prg        = nullptr;

    // マテリアル用のCallableプログラム
    // OptiXでは基底クラスのポインタを介した、派生クラスの関数呼び出し (ポリモーフィズム)が
    // 禁止されているため、Callable関数を使って疑似的なポリモーフィズムを実現する
    // ここでは、Lambertian, Dielectric, Metal の3種類を実装している
    CallableProgram             lambertian_prg           = {};
    CallableProgram             dielectric_prg           = {};
    CallableProgram             metal_prg                = {};

    // テクスチャ用のCallableプログラム
    // Constant ... 単色、Checker ... チェッカーボード
    CallableProgram             constant_prg             = {};
    CallableProgram             checker_prg              = {};

    // CUDA stream
    CUstream                    stream                   = 0;

    // Pipeline launch parameters
    // CUDA内で extern "C" __constant__ Params params
    // と宣言することで、全モジュールからアクセス可能である。
    Params                      params;
    Params*                     d_params;

    // Shader binding table
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
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
    // 右クリック中にマウスが移動した場合は、カメラの原点を固定して注視点を動かす
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
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
    // 球体のAxis-aligned bounding box (AABB)を返す
    const float3 center = sphere.center;
    const float radius = sphere.radius;
    return OptixAabb {
        /* minX = */ center.x - radius, /* minY = */ center.y - radius, /* minZ = */ center.z - radius, 
        /* maxX = */ center.x + radius, /* maxY = */ center.y + radius, /* maxZ = */ center.z + radius
    };
}

// -----------------------------------------------------------------------
void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}

// -----------------------------------------------------------------------
// Pipeline launch parameterの初期化
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
    state.params.max_depth = 5;

    // レイトレーシングを行うASのtraversableHandleを設定
    state.params.handle = state.ias.handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

// -----------------------------------------------------------------------
// カメラの更新処理
// マウス入力等でカメラが動いた際にlaunch parameterも更新する
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
// ウィンドウサイズが変化したときの処理
// レイトレーシングによる計算結果を格納するバッファを更新する
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
// カメラとウィンドウサイズの変化を監視
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
// optixLaunchを呼び出し、デバイス側のレイトレーシングカーネルを起動
// -----------------------------------------------------------------------
void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, OneWeekendState& state )
{
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
// OpenGLを介してレンダリング結果を描画
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
// デバイス側からのメッセージを取得するためのCallable関数
// OptixDeviceContextを生成する際に、
// OptixDeviceContext::logCallbackFunctionに登録する
// -----------------------------------------------------------------------
static void contextLogCallback(uint32_t level, const char* tag, const char* msg, void* /* callback_data */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << msg << "\n";
}

// -----------------------------------------------------------------------
// カメラの初期化
// -----------------------------------------------------------------------
void initCameraState()
{
    camera_changed = true;

    camera.setEye(make_float3(13.0f, 2.0f, 3.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(20.0f);
    camera.setAspectRatio(3.0f / 2.0f);

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}

// -----------------------------------------------------------------------
// OptixDeviceContextの初期化
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
    // Callbackで取得するメッセージのレベル
    // 0 ... disable、メッセージを受け取らない
    // 1 ... fatal、修復不可能なエラー。コンテクストやOptiXが不能状態にある
    // 2 ... error、修復可能エラー。
    // 3 ... warning、意図せぬ挙動や低パフォーマンスを導くような場合に警告してくれる
    // 4 ... print、全メッセージを受け取る
    options.logCallbackLevel     = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

// -----------------------------------------------------------------------
// 重複のないインデックスの個数を数える
// 例) { 0, 0, 0, 1, 1, 2, 2, 2 } -> 3 
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
// Geometry acceleration structureの構築
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

        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas.handle, gas.d_output_buffer, compacted_gas_size, &gas.handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// -----------------------------------------------------------------------
// Mesh用のGASを構築
// デバイス側のポインタ(state.d_mesh_data)へのデータコピーも同時に行う
// -----------------------------------------------------------------------
void buildMeshGAS(
    OneWeekendState& state, 
    GeometryAccelData& gas,
    const std::vector<float3>& vertices, 
    const std::vector<uint3>& indices, 
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
    const size_t indices_size = indices.size() * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(), indices_size, 
        cudaMemcpyHostToDevice 
    ));
    
    MeshData mesh_data{reinterpret_cast<float3*>(d_vertices), reinterpret_cast<uint3*>(d_indices) };
    CUDA_CHECK(cudaMalloc(&state.d_mesh_data, sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        state.d_mesh_data, &mesh_data, sizeof(MeshData), cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_sbt_indices = 0;
    const size_t sbt_indices_size = sbt_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sbt_indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void**>(d_sbt_indices),
        sbt_indices.data(), sbt_indices_size,
        cudaMemcpyHostToDevice
    ));

    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

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
    mesh_input.triangleArray.indexBuffer = d_indices;
    mesh_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
    mesh_input.triangleArray.numSbtRecords = num_sbt_records;
    mesh_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
    mesh_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    mesh_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(state, gas, mesh_input);
}

// -----------------------------------------------------------------------
// Sphere用のGASを構築
// デバイス側のポインタ(state.d_sphere_data)へのデータコピーも同時に行う
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

    CUDA_CHECK(cudaMalloc(&state.d_sphere_data, sizeof(SphereData) * spheres.size()));
    CUDA_CHECK(cudaMemcpy(
        state.d_sphere_data, spheres.data(), sizeof(SphereData) * spheres.size(), cudaMemcpyHostToDevice
    ));

    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

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

// -----------------------------------------------------------------------
// Instance acceleration structureの構築
// -----------------------------------------------------------------------
void buildIAS(OneWeekendState& state, InstanceAccelData& ias, const std::vector<OptixInstance>& instances)
{
    CUdeviceptr d_instances;
    const size_t instances_size = sizeof(OptixInstance) * instances.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        instances.data(), instances_size,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &instance_input,
        1, // num build input
        &ias_buffer_sizes
    ));

    size_t d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;

    // Allocate buffer to build acceleration structure
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        d_temp_buffer_size
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&ias.d_output_buffer),
        ias_buffer_sizes.outputSizeInBytes
    ));

    // Build instance AS contains all GASs to describe the scene
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        d_temp_buffer,
        d_temp_buffer_size,
        ias.d_output_buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &ias.handle,        // emitted property list
        nullptr,            // num emitted property
        0
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
}

// -----------------------------------------------------------------------
// OptixModuleの作成
// -----------------------------------------------------------------------
void createModule(OneWeekendState& state)
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues = 2;
    state.pipeline_compile_options.numAttributeValues = 5;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixInOneWeekend.cu", inputSize);

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.module
    ));
}

// -----------------------------------------------------------------------
// 全ProgramGroupの作成
// -----------------------------------------------------------------------
void createProgramGroups(OneWeekendState& state)
{
    OptixProgramGroupOptions prg_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Raygen program
    {
        OptixProgramGroupDesc raygen_prg_desc = {};
        raygen_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prg_desc.raygen.module = state.module;
        raygen_prg_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, 
            &raygen_prg_desc, 
            1, // num program groups
            &prg_options, 
            log, 
            &sizeof_log, 
            &state.raygen_prg
        ));
    }

    // Miss program
    {
        OptixProgramGroupDesc miss_prg_desc = {};
        miss_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prg_desc.miss.module = state.module;
        miss_prg_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, 
            &miss_prg_desc, 
            1, 
            &prg_options, 
            log, 
            &sizeof_log, 
            &state.miss_prg
        ));
    }

    // Hitgroup programs
    {
        // Mesh
        OptixProgramGroupDesc hitgroup_prg_desc = {};
        hitgroup_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prg_desc.hitgroup.moduleCH = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameCH = "__closesthit__mesh";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_prg_desc,
            1,
            &prg_options,
            log,
            &sizeof_log,
            &state.mesh_hitgroup_prg
        ));

        // Sphere
        memset(&hitgroup_prg_desc, 0, sizeof(OptixProgramGroupDesc));
        hitgroup_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prg_desc.hitgroup.moduleIS = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        hitgroup_prg_desc.hitgroup.moduleCH = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_prg_desc,
            1,
            &prg_options,
            log,
            &sizeof_log,
            &state.sphere_hitgroup_prg
        ));
    }

    uint32_t callables_id = 0;
    auto createDirectCallables = [&](CallableProgram& callable, const char* dc_function_name)
    {
        OptixProgramGroupDesc callables_prg_desc = {};

        callables_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callables_prg_desc.callables.moduleDC = state.module;
        callables_prg_desc.callables.entryFunctionNameDC = dc_function_name;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &callables_prg_desc,
            1,
            &prg_options,
            log,
            &sizeof_log,
            &callable.program
        ));
        callable.id = callables_id;
        callables_id++;
    };

    // マテリアル用のCallableプログラム
    {
        // Lambertian
        createDirectCallables(state.lambertian_prg, "__direct_callable__lambertian");
        // Dielectric
        createDirectCallables(state.dielectric_prg, "__direct_callable__dielectric");
        // Metal
        createDirectCallables(state.metal_prg, "__direct_callable__metal");
    }

    // テクスチャ用のCallableプログラム
    {
        // Constant texture
        createDirectCallables(state.constant_prg, "__direct_callable__constant");
        // Checker texture
        createDirectCallables(state.checker_prg, "__direct_callable__checker");
    }
}

// -----------------------------------------------------------------------
// OptixPipelineの作成
// -----------------------------------------------------------------------
void createPipeline(OneWeekendState& state)
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prg, 
        state.miss_prg, 
        state.mesh_hitgroup_prg, 
        state.sphere_hitgroup_prg, 
        state.lambertian_prg.program, 
        state.dielectric_prg.program,
        state.metal_prg.program, 
        state.constant_prg.program, 
        state.checker_prg.program
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    // optixTrace()の呼び出し深度の設定
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &state.pipeline
    ));

    // 各プログラムからパイプラインによって構築されるCall graphのスタックサイズを計算
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.miss_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.mesh_hitgroup_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.sphere_hitgroup_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.lambertian_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.dielectric_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.metal_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.constant_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.checker_prg.program, &stack_sizes));

    uint32_t max_trace_depth = pipeline_link_options.maxTraceDepth;
    // Continuation callableは使用していないので、0でよい
    uint32_t max_cc_depth = 0;
    // Direct callableの呼び出し深度は最大でも2回 (マテリアル -> テクスチャ)
    uint32_t max_dc_depth = 3;
    uint32_t direct_callable_stack_size_from_traversable;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, 
        max_trace_depth, 
        max_cc_depth, 
        max_dc_depth,
        &direct_callable_stack_size_from_traversable,
        &direct_callable_stack_size_from_state, 
        &continuation_stack_size
    ));

    // Traversable graphの深度を設定する
    // 今回のように IAS -> GAS だけで終わるのであれば、traversable graphの深度は2となる
    // IAS -> Motion transform -> GAS となるようであれば、深度は3必要となる
    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline, 
        direct_callable_stack_size_from_traversable, 
        direct_callable_stack_size_from_state,
        continuation_stack_size, 
        max_traversal_depth
    ));
}

// -----------------------------------------------------------------------
// Shader binding tableの構築
// -----------------------------------------------------------------------
void createSBT(OneWeekendState& state, const std::vector<std::pair<ShapeType, HitGroupData>>& hitgroup_datas)
{
    // Ray generation 
    RayGenRecord raygen_record = {};
    // RayGenRecordの領域をデバイス側に確保
    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenRecord)));
    // SBT recordのヘッダーをプログラムを使って埋める
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prg, &raygen_record));
    // RayGenRecordをデバイス側にコピー
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &raygen_record,
        sizeof(RayGenRecord),
        cudaMemcpyHostToDevice
    ));

    // Miss
    MissRecord miss_record = {};
    // MissRecordの領域をデバイス側に確保
    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissRecord)));
    // SBT recordのヘッダーをプログラムを使って埋める
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prg, &miss_record));
    // データを設定
    miss_record.data.bg_color = make_float4(0.0f);
    // MissRecordをデバイス側にコピー
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        &miss_record,
        sizeof(MissRecord),
        cudaMemcpyHostToDevice
    ));

    // HitGroup
    HitGroupRecord* hitgroup_records = new HitGroupRecord[hitgroup_datas.size()];
    // HitGroupRecord用の領域をデバイス側に確保
    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord) * hitgroup_datas.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hitgroup_record_size));

    // HitGroupDataからShader binding tableを構築
    for (size_t i = 0; i < hitgroup_datas.size(); i++)
    {
        ShapeType type = hitgroup_datas[i].first;
        HitGroupData data = hitgroup_datas[i].second;
        // ShapeTypeに応じてヘッダーを埋めるためのプログラムを切り替える
        if (type == ShapeType::Mesh)
            OPTIX_CHECK(optixSbtRecordPackHeader(state.mesh_hitgroup_prg, &hitgroup_records[i]));
        else if (type == ShapeType::Sphere)
            OPTIX_CHECK(optixSbtRecordPackHeader(state.sphere_hitgroup_prg, &hitgroup_records[i]));
        // データを設定
        hitgroup_records[i].data = data;
    }
    // HitGroupRecordをデバイス側にコピー
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    // 今回はCallable プログラムのShader binding tableへのデータ登録は必要ないので、
    // EmptyRecordを使って空データをコピーする。
    // ただし、データがない場合でもヘッダーをプログラムで埋める必要がある。
    // ここを忘れるとレイトレーシング起動後にInvalid memory accessが起きる
    // デバッグで気づきづらい点なので要注意
    EmptyRecord* callables_records = new EmptyRecord[5];
    CUdeviceptr d_callables_records;
    const size_t callables_record_size = sizeof(EmptyRecord) * 5;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callables_records), callables_record_size));

    OPTIX_CHECK(optixSbtRecordPackHeader(state.lambertian_prg.program, &callables_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.dielectric_prg.program, &callables_records[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.metal_prg.program, &callables_records[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.constant_prg.program, &callables_records[3]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.checker_prg.program, &callables_records[4]));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_callables_records),
        callables_records,
        callables_record_size,
        cudaMemcpyHostToDevice
    ));

    // 各recordからShader binding tableを構築
    // ここではrecord配列の先頭へのポインタと、shader binding tableのアラインメント、配列数を設定する
    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_record;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupRecord));
    state.sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_datas.size());
    state.sbt.callablesRecordBase = d_callables_records;
    state.sbt.callablesRecordCount = 5;
    state.sbt.callablesRecordStrideInBytes = sizeof(EmptyRecord);
}

// -----------------------------------------------------------------------
void finalizeState(OneWeekendState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.mesh_hitgroup_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.sphere_hitgroup_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.lambertian_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.dielectric_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.metal_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.constant_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.checker_prg.program));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
}

// -----------------------------------------------------------------------
void createScene(OneWeekendState& state)
{
    // デバイス上にデータをコピーして、そのポインタを汎用ポインタで返すlambda関数
    auto copyDataToDevice = [](auto data, size_t size) -> void*
    {
        CUdeviceptr device_ptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(device_ptr),
            &data, size,
            cudaMemcpyHostToDevice
        ));
        return reinterpret_cast<void*>(device_ptr);
    };

    // HitGroupDataとマテリアルデータを格納する配列
    // 今回の場合は、球・メッシュではそれぞれでジオメトリ用のデータは同じ配列を使用し、
    // デバイス側でのoptixGetPrimitiveIndex()で交差するデータを切り替えて
    // マテリアルデータは異なるデータが振り分けられている方式をとっている。
    // そのため、hitgroup_datasの数はmaterialsの数に合わせる 
    // <- マテリアルの分だけHitGroupRecordがあれば十分でジオメトリの数用意する必要はない
    std::vector<std::pair<ShapeType, HitGroupData>> hitgroup_datas;
    std::vector<Material> materials;

    // --------------------------------------------------------------------
    // 球体のシーン構築
    // 球体は全て異なるマテリアルを持っていることとする
    // --------------------------------------------------------------------
    // 球体用のデータ準備
    std::vector<SphereData> spheres;
    // 球体用の相対的なsbt_indexの配列
    std::vector<uint32_t> sphere_sbt_indices;
    uint32_t sphere_sbt_index = 0;

    // Ground
    SphereData ground_sphere{ make_float3(0, -1000, 0), 1000 };
    spheres.emplace_back(ground_sphere);
    // テクスチャ
    CheckerData ground_checker{ make_float4(1.0f), make_float4(0.2f, 0.5f, 0.2f, 1.0f), 5000};
    // Lambertianマテリアル
    LambertianData ground_lambert{ copyDataToDevice(ground_checker, sizeof(CheckerData)), state.checker_prg.id };
    materials.push_back(Material{ copyDataToDevice(ground_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    // マテリアルを追加したのでsbt_indexも追加
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);
    
    // 疑似乱数用のシード値を生成
    uint32_t seed = tea<4>(0, 0);
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            const float choose_mat = rnd(seed);
            const float3 center{ a + 0.9f * rnd(seed), 0.2f, b + 0.9f * rnd(seed) };
            if (length(center - make_float3(4, 0.2, 0)) > 0.9f)
            {
                // 球体を追加
                spheres.emplace_back( SphereData { center, 0.2f });

                // 確率的にLambertian、Metal、Dielectricマテリアルを作成
                // 追加する際は型に応じたCallableプログラムIDを割り振る
                if (choose_mat < 0.8f)
                {
                    // Lambertian
                    ConstantData albedo{ make_float4(rnd(seed), rnd(seed), rnd(seed), 1.0f) };
                    LambertianData lambertian{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id };
                    materials.emplace_back(Material{ copyDataToDevice(lambertian, sizeof(LambertianData)), state.lambertian_prg.id });
                }
                else if (choose_mat < 0.95f)
                {
                    // Metal
                    ConstantData albedo{ make_float4(0.5f + rnd(seed) * 0.5f) };
                    MetalData metal{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id, /* fuzz = */ rnd(seed) * 0.5f};
                    materials.emplace_back(Material{ copyDataToDevice(metal, sizeof(MetalData)), state.metal_prg.id });
                }
                else
                {
                    // Dielectric
                    ConstantData albedo{ make_float4(1.0f) };
                    DielectricData glass{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id, /* ior = */ 1.5f};
                    materials.emplace_back(Material{ copyDataToDevice(glass, sizeof(DielectricData)), state.dielectric_prg.id });
                }
                sphere_sbt_indices.emplace_back(sphere_sbt_index++);
            }
        }
    }
    
    // Dielectric
    spheres.emplace_back(SphereData{ make_float3(0.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo1{ make_float4(1.0f) };
    DielectricData material1{ copyDataToDevice(albedo1, sizeof(ConstantData)), state.constant_prg.id, /* ior = */ 1.5f };
    materials.push_back(Material{ copyDataToDevice(material1, sizeof(DielectricData)), state.dielectric_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Lambertian
    spheres.emplace_back(SphereData{ make_float3(-4.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo2{ make_float4(0.4f, 0.2f, 0.1f, 1.0f) };
    LambertianData material2{ copyDataToDevice(albedo2, sizeof(ConstantData)), state.constant_prg.id };
    materials.push_back(Material{ copyDataToDevice(material2, sizeof(LambertianData)), state.lambertian_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Metal
    spheres.emplace_back(SphereData{ make_float3(4.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo3{ make_float4(0.7f, 0.6f, 0.5f, 1.0f) };
    MetalData material3{ copyDataToDevice(albedo3, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(material3, sizeof(MetalData)), state.metal_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Sphere用のGASを作成 (内部で同時にstate.d_sphere_dataへのデータコピーも行っている)
    GeometryAccelData sphere_gas;
    buildSphereGAS(state, sphere_gas, spheres, sphere_sbt_indices);

    // マテリアルと球体データの配列からShader binding table用のデータを用意
    for (auto& m : materials)
        hitgroup_datas.emplace_back(ShapeType::Sphere, HitGroupData{state.d_sphere_data, m});

    // --------------------------------------------------------------------
    // メッシュののシーン構築
    // メッシュでは100個の三角形に対して割り振るマテリアルは3種類のみ
    // メッシュデータは全マテリアル共通なので、用意するSBT recordも3つのみでよい
    // --------------------------------------------------------------------
    std::vector<float3> mesh_vertices;
    std::vector<uint3> mesh_indices;
    std::vector<uint32_t> mesh_sbt_indices;
    uint32_t mesh_index = 0;
    for (int a = 0; a < 100; a++) {
        float3 center{rnd(seed) * 20.0f - 10.0f, 0.5f + rnd(seed) * 1.0f - 0.5f, rnd(seed) * 20.0f - 10.0f };
        const float3 p0 = center + make_float3(rnd(seed) * 0.5f, -rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);
        const float3 p1 = center + make_float3(-rnd(seed) * 0.5f, -rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);
        const float3 p2 = center + make_float3(rnd(seed) * 0.25f, rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);

        mesh_vertices.emplace_back(p0);
        mesh_vertices.emplace_back(p1);
        mesh_vertices.emplace_back(p2);
        mesh_indices.emplace_back(make_uint3(mesh_index + 0, mesh_index + 1, mesh_index + 2));
        mesh_index += 3;
    }

    const uint32_t red_sbt_index = 0;
    const uint32_t green_sbt_index = 1;
    const uint32_t blue_sbt_index = 2;

    // ランダムで赤・緑・青の3色を割り振る
    for (const auto& face : mesh_indices)
    {
        const float choose_rgb = rnd(seed);
        if (choose_rgb < 0.33f)
            mesh_sbt_indices.push_back(red_sbt_index);
        else if (choose_rgb < 0.67f)
            mesh_sbt_indices.push_back(green_sbt_index);
        else
            mesh_sbt_indices.push_back(blue_sbt_index);
    }

    // メッシュ用のGASを作成
    GeometryAccelData mesh_gas;
    buildMeshGAS(state, mesh_gas, mesh_vertices, mesh_indices, mesh_sbt_indices);

    // 赤・緑・青のマテリアルを用意し、HitGroupDataを追加
    // 赤
    ConstantData red{ {0.8f, 0.05f, 0.05f, 1.0f} };
    LambertianData red_lambert{ copyDataToDevice(red, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(red_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // 緑
    ConstantData green{ {0.05f, 0.8f, 0.05f, 1.0f} };
    LambertianData green_lambert{ copyDataToDevice(green, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(green_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // 青
    ConstantData blue{ {0.05f, 0.05f, 0.8f, 1.0f} };
    LambertianData blue_lambert{ copyDataToDevice(blue, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(blue_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // IAS用のInstanceを球体用・メッシュ用それぞれ作成
    std::vector<OptixInstance> instances;
    uint32_t flags = OPTIX_INSTANCE_FLAG_NONE;

    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;
    instances.emplace_back(OptixInstance{
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, instance_id, sbt_offset, 255, 
        flags, sphere_gas.handle, {0, 0}
    });

    sbt_offset += sphere_gas.num_sbt_records;
    instance_id++;
    // メッシュの方はY軸中心にPI/6だけ回転させる
    const float c = cosf(M_PIf / 6.0f);
    const float s = sinf(M_PIf / 6.0f);
    instances.push_back(OptixInstance{
        {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0}, instance_id, sbt_offset, 255,
        flags, mesh_gas.handle, {0, 0}
    });

    // IASの作成
    buildIAS(state, state.ias, instances);

    // Shader binding tableの作成
    createSBT(state, hitgroup_datas);
}

// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    OneWeekendState state;
    state.params.width = 1200;
    state.params.height = static_cast<int>(1200.0f / (3.0f / 2.0f));
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    std::string outfile;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        initCameraState();

        createContext(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createScene(state);
        initLaunchParams(state);

        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixInOneWeekend", state.params.width, state.params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &state.params);

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            handleCameraUpdate(state.params);
            handleResize(output_buffer, state.params);
            for (int i = 0; i < 1024; i += samples_per_launch) {
                launchSubframe(output_buffer, state);
                state.params.subframe_index++;
            }

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        finalizeState(state);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}