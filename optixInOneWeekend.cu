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
#include <optix.h>

#include "optixInOneWeekend.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    float3       normal; 
    float2       texcoord;

    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;

    // マテリアル用のデータとCallablesプログラムのID
    Material material;
};

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}

static __forceinline__ __device__ float3 randomInUnitSphere(unsigned int& seed) {
    const float phi = 2.0f * M_PIf * rnd(seed);
    const float theta = acosf(1.0f - 2.0f * rnd(seed));
    const float x = sinf(theta) * cosf(phi);
    const float y = sinf(theta) * sinf(phi);
    const float z = cosf(theta);
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 randomSampleHemisphere(unsigned int& seed, const float3& normal)
{
    const float3 vec_in_sphere = randomInUnitSphere(seed);
    if (dot(vec_in_sphere, normal) > 0.0f)
        return vec_in_sphere;
    else
        return -vec_in_sphere;
}

static __forceinline__ __device__ float3 cosineSampleHemisphere(const float u1, const float u2)
{
    const float r = sqrtf(u2);
    const float phi = 2.0f * M_PIf * u1;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(1.0f - u2);
    return make_float3(x, y, z);
}


static __forceinline__ __device__ float fresnel(float cosine, float ref_idx)
{
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5.0f);
}

static __forceinline__ __device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0f);
    float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3 r_out_parallel = -sqrtf(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

static __forceinline__ __device__ float3 refract(const float3& wi, const float3& n, float cos_i, float ni, float nt) {
    float nt_ni = nt / ni;
    float ni_nt = ni / nt;
    float D = sqrtf(nt_ni * nt_ni - (1.0f - cos_i * cos_i)) - cos_i;
    return ni_nt * (wi - D * n);
}

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,        // SBT offset
            1,        // SBT stride
            0,        // missSBTIndex
            u0, u1 );
}

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const int w = params.width; 
    const int h = params.height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V; 
    const float3 W = params.W;
    const uint3 idx = optixGetLaunchIndex();
    const int subframe_index = params.subframe_index;
    const int samples_per_launch = params.samples_per_launch;

    // 現在のスレッドIDから乱数用のシード値を生成
    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < samples_per_launch; i++)
    {
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
            ((float)idx.x + subpixel_jitter.x) / (float)w, 
            ((float)idx.y + subpixel_jitter.y) / (float)h
        ) - 1.0f;

        // 光線の向きと原点を設定
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.emitted = make_float3(0.0f);
        prd.radiance = make_float3(0.0f);
        prd.attenuation = make_float3(1.0f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;

        float3 throughput = make_float3(1.0f);

        int depth = 0;
        for (;;)
        {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ray_origin, ray_direction, 0.01f, 1e16f, &prd);

            if (prd.done) {
                result += prd.emitted * throughput;
                break;
            }

            // Direct callable関数を使って各マテリアルにおける
            optixDirectCall<void, RadiancePRD*, void*>(
                prd.material.prg_id, &prd, prd.material.data
            );

            throughput *= prd.attenuation;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    }

    const unsigned int image_index = idx.y * params.width + idx.x;
    float3 accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);
}

extern "C" __global__ void __miss__radiance()
{
    RadiancePRD* prd = getPRD();
    const float3 unit_direction = normalize(optixGetWorldRayDirection());
    const float t = 0.5f * (unit_direction.y + 1.0f);
    prd->emitted = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
    prd->done      = true;
}

extern "C" __global__ void __closesthit__mesh()
{
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const MeshData* mesh_data = (MeshData*)data->shape_data;

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 direction         = optixGetWorldRayDirection();
    const uint3 index = mesh_data->indices[prim_idx];

    const float2 texcoord = optixGetTriangleBarycentrics();

    const float3 v0   = mesh_data->vertices[ index.x ];
    const float3 v1   = mesh_data->vertices[ index.y ];
    const float3 v2   = mesh_data->vertices[ index.z ];
    const float3 N  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*direction;

    RadiancePRD* prd = getPRD();

    prd->origin = P;
    prd->direction = direction;
    prd->normal = N;
    prd->texcoord = texcoord;
    prd->material = data->material;
}

extern "C" __global__ void __intersection__sphere()
{
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const int prim_idx = optixGetPrimitiveIndex();
    const SphereData sphere_data = ((SphereData*)data->shape_data)[prim_idx];

    const float3 center = sphere_data.center;
    const float radius = sphere_data.radius;

    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    const float3 oc = origin - center;
    const float a = dot(direction, direction);
    const float half_b = dot(oc, direction);
    const float c = dot(oc, oc) - radius * radius;

    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return;
    
    const float sqrtd = sqrtf(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < tmin || tmax < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < tmin || tmax < root)
            return;
    }

    const float3 P = origin + root * direction;
    const float3 normal = (P - center) / radius;

    float phi = atan2(normal.y, normal.x);
    if (phi < 0) phi += 2.0f * M_PIf;
    const float theta = acosf(normal.z);
    const float2 texcoord = make_float2(phi / (2.0f * M_PIf), theta / M_PIf);

    optixReportIntersection(root, 0, 
        __float_as_int(normal.x), __float_as_int(normal.y), __float_as_int(normal.z),
        __float_as_int(texcoord.x), __float_as_int(texcoord.y)
    );
}

extern "C" __global__ void __closesthit__sphere()
{
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

    const float3 local_n = make_float3(
        __int_as_float(optixGetAttribute_0()),
        __int_as_float(optixGetAttribute_1()),
        __int_as_float(optixGetAttribute_2())
    );
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    const float2 texcoord = make_float2(
        __int_as_float(optixGetAttribute_3()),
        __int_as_float(optixGetAttribute_4())
    );

    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();
    const float3 P = origin + optixGetRayTmax() * direction;

    RadiancePRD* prd = getPRD();
    prd->origin = P;
    prd->normal = world_n;
    prd->direction = direction;
    prd->texcoord = texcoord;
    prd->material = data->material;
}

extern "C" __device__ void __direct_callable__lambertian(RadiancePRD* prd, void* material_data)
{
    const LambertianData* lambertian = (LambertianData*)material_data;
    const float4 color = optixDirectCall<float4, RadiancePRD*, void*>(
        lambertian->texture_prg_id, prd, lambertian->texture_data
        );
    prd->attenuation = make_float3(color);

    prd->normal = faceforward(prd->normal, -prd->direction, prd->normal);

    unsigned int seed = prd->seed;
    float3 wi = randomSampleHemisphere(seed, prd->normal);
    prd->direction = normalize(wi);
    prd->done = false;
    prd->emitted = make_float3(0.0f);
}

extern "C" __device__ void __direct_callable__dielectric(RadiancePRD* prd, void* material_data)
{
    const DielectricData* dielectric = (DielectricData*)material_data;
    const float4 color = optixDirectCall<float4, RadiancePRD*, void*>(
        dielectric->texture_prg_id, prd, dielectric->texture_data
        );

    const float ior = dielectric->ior;
    const float3 in_direction = prd->direction;

    prd->attenuation = make_float3(color);
    float cos_theta = dot(in_direction, prd->normal);
    bool into = cos_theta < 0;
    const float3 outward_normal = into ? prd->normal : -prd->normal;
    const float refraction_ratio = into ? (1.0 / ior) : ior;

    float3 unit_direction = normalize(in_direction);
    cos_theta = fabs(cos_theta);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    unsigned int seed = prd->seed;
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    if (cannot_refract || rnd(seed) < fresnel(cos_theta, refraction_ratio))
        prd->direction = reflect(unit_direction, prd->normal);
    else
        prd->direction = refract(unit_direction, outward_normal, refraction_ratio);
    prd->done = false;
    prd->emitted = make_float3(0.0f);
    prd->seed = seed;
}

extern "C" __device__ void __direct_callable__metal(RadiancePRD* prd, void* material_data)
{
    const MetalData* metal = (MetalData*)material_data;
    unsigned int seed = prd->seed;
    prd->direction = reflect(prd->direction, prd->normal) + metal->fuzz * randomInUnitSphere(seed);
    const float4 color = optixDirectCall<float4, RadiancePRD*, void*>(
        metal->texture_prg_id, prd, metal->texture_data
        );
    prd->attenuation = make_float3(color);
    prd->done = false;
    prd->emitted = make_float3(0.0f);
    prd->seed = seed;
}

extern "C" __device__ float4 __direct_callable__constant(RadiancePRD* /* prd */ , void* texture_data)
{
    const ConstantData* constant = (ConstantData*)texture_data;
    return constant->color;
}

extern "C" __device__ float4 __direct_callable__checker(RadiancePRD* prd, void* texture_data)
{
    const CheckerData* checker = (CheckerData*)texture_data;
    const bool is_odd = sinf(prd->texcoord.x * M_PIf * checker->scale) * sinf(prd->texcoord.y * M_PIf * checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;

}