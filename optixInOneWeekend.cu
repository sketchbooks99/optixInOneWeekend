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

struct SurfaceInfo
{
    // 発光度
    float3 emission;
    // 物体表面の色
    float3 albedo;
    // 衝突位置
    float3 p;
    // レイの方向
    float3 direction;
    // 法線
    float3 n;
    // テクスチャ座標
    float2 texcoord;

    // 乱数のシード値
    unsigned int seed;
    // 光線追跡を終了するか否か
    int trace_terminate;

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

// ポインタをunsigned long longに変換してから、前側32bitをi0に、後側32bitをi1に格納する
static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

// 0番目と1番目のペイロードにパックされているSurfaceInfoのポインタを取得
static __forceinline__ __device__ SurfaceInfo* getSurfaceInfo()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInfo*>( unpackPointer( u0, u1 ) );
}

static __forceinline__ __device__ float3 randomInUnitSphere(unsigned int& seed) {
    while (true)
    {
        float3 v = make_float3(rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f);
        if (dot(v, v) >= 1.0f) continue;
        return v;
    }
}

static __forceinline__ __device__ float3 randomSampleHemisphere(unsigned int& seed, const float3& normal)
{
    const float3 vec_in_sphere = randomInUnitSphere(seed);
    if (dot(vec_in_sphere, normal) > 0.0f)
        return vec_in_sphere;
    else
        return -vec_in_sphere;
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
        SurfaceInfo*           si
        )
{
    // SurfaceInfoのポインタを2つのペイロードにパックする
    unsigned int u0, u1;
    packPointer( si, u0, u1 );
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

        SurfaceInfo si;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.trace_terminate = false;
        si.seed = seed;

        float3 throughput = make_float3(1.0f);

        int depth = 0;
        for (;;)
        {
            if (depth >= params.max_depth)
                break;

            // IASに対してレイトレース
            trace(params.handle, ray_origin, ray_direction, 0.01f, 1e16f, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            // Direct callable関数を使って各マテリアルにおける散乱方向とマテリアルの色を計算
            float3 scattered;
            optixDirectCall<void, SurfaceInfo*, void*, float3&>(
                si.material.prg_id, &si, si.material.data, scattered
            );

            throughput *= si.albedo;

            ray_origin = si.p;
            ray_direction = scattered;

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
    // 取得した輝度値を出力バッファに書き込む
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);
}

extern "C" __global__ void __miss__radiance()
{
    const MissData* miss = (MissData*)optixGetSbtDataPointer();

    SurfaceInfo* si = getSurfaceInfo();

    // ベクトルのy成分から背景色を計算
    const float3 unit_direction = normalize(optixGetWorldRayDirection());
    const float t = 0.5f * (unit_direction.y + 1.0f);
    si->emission = (1.0f - t) * make_float3(1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
    si->trace_terminate      = true;
}

extern "C" __global__ void __closesthit__mesh()
{
    // Shader binding tableからデータを取得
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    const MeshData* mesh_data = (MeshData*)data->shape_data;

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 direction         = optixGetWorldRayDirection();
    const uint3 index = mesh_data->indices[prim_idx];

    // 三角形の重心座標(u,v)を三角形のテクスチャ座標とする
    const float2 texcoord = optixGetTriangleBarycentrics();

    // メッシュデータから頂点を取得し、法線計算
    const float3 v0   = mesh_data->vertices[ index.x ];
    const float3 v1   = mesh_data->vertices[ index.y ];
    const float3 v2   = mesh_data->vertices[ index.z ];
    const float3 N  = normalize( cross( v1-v0, v2-v0 ) );

    // レイと三角形の交点を計算
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*direction;

    // PayloadからSurfaceInfoのポインタを取得し、交点上の情報を格納
    SurfaceInfo* si = getSurfaceInfo();

    // SurfaceInfoに交点における情報を格納する
    si->p = P;
    si->direction = direction;
    si->n = N;
    si->texcoord = texcoord;
    // HitGroupDataに紐付いているマテリアル情報をSurfaceInfoに紐付ける
    si->material = data->material;
}

extern "C" __global__ void __intersection__sphere()
{
    // Shader binding tableからデータを取得
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
    // AABBとの交差判定が認められた球体のGAS内のIDを取得
    const int prim_idx = optixGetPrimitiveIndex();
    const SphereData sphere_data = ((SphereData*)data->shape_data)[prim_idx];

    const float3 center = sphere_data.center;
    const float radius = sphere_data.radius;

    // オブジェクト空間におけるレイの原点と方向を取得
    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();
    // レイの最小距離と最大距離を取得
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    // 球体との交差判定処理（判別式を解いて、距離tを計算)
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

    // オブジェクト空間におけるレイと球の交点を計算
    const float3 P = origin + root * direction;
    const float3 normal = (P - center) / radius;

    // 球体におけるテクスチャ座標を算出 (Z up)と仮定して、xとyから方位角、zから仰角を計算
    float phi = atan2(normal.y, normal.x);
    if (phi < 0) phi += 2.0f * M_PIf;
    const float theta = acosf(normal.z);
    const float2 texcoord = make_float2(phi / (2.0f * M_PIf), theta / M_PIf);

    // レイと球の交差判定を認める
    optixReportIntersection(root, 0, 
        __float_as_int(normal.x), __float_as_int(normal.y), __float_as_int(normal.z),
        __float_as_int(texcoord.x), __float_as_int(texcoord.y)
    );
}

extern "C" __global__ void __closesthit__sphere()
{
    // Shader binding tableからデータを取得
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

    // 0 - 2番目のAttributeからIntersectionプログラムで計算した法線を取得
    const float3 local_n = make_float3(
        __int_as_float(optixGetAttribute_0()),
        __int_as_float(optixGetAttribute_1()),
        __int_as_float(optixGetAttribute_2())
    );
    // Instanceに紐付いている行列からオブジェクト空間における法線をグローバル空間にマップする
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    // 3 - 4番目のAttributeからテクスチャ座標を取得
    const float2 texcoord = make_float2(
        __int_as_float(optixGetAttribute_3()),
        __int_as_float(optixGetAttribute_4())
    );

    // グローバル空間におけるレイの原点と方向を計算し、交点座標の位置を計算
    const float3 origin = optixGetWorldRayOrigin();
    const float3 direction = optixGetWorldRayDirection();
    const float3 P = origin + optixGetRayTmax() * direction;

    // PayloadからSurfaceInfoのポインタを取得し、交点上の情報を格納
    SurfaceInfo* si = getSurfaceInfo();
    si->p = P;
    si->n = world_n;
    si->direction = direction;
    si->texcoord = texcoord;
    // HitGroupDataに紐付いているマテリアル情報をSurfaceInfoに紐付ける
    si->material = data->material;
}

extern "C" __device__ void __direct_callable__lambertian(SurfaceInfo* si, void* material_data, float3& scattered)
{
    const LambertianData* lambertian = (LambertianData*)material_data;

    // Direct callableプログラムによって、テクスチャ色を取得
    const float4 color = optixDirectCall<float4, SurfaceInfo*, void*>(
        lambertian->texture_prg_id, si, lambertian->texture_data
        );
    si->albedo = make_float3(color);

    si->n = faceforward(si->n, -si->direction, si->n);

    unsigned int seed = si->seed;
    float3 wi = randomSampleHemisphere(seed, si->n);
    scattered = normalize(wi);
    si->trace_terminate = false;
    si->emission = make_float3(0.0f);
}

extern "C" __device__ void __direct_callable__dielectric(SurfaceInfo* si, void* material_data, float3& scattered)
{
    const DielectricData* dielectric = (DielectricData*)material_data;
    // Direct callableプログラムによって、テクスチャ色を取得
    const float4 color = optixDirectCall<float4, SurfaceInfo*, void*>(
        dielectric->texture_prg_id, si, dielectric->texture_data
        );

    const float ior = dielectric->ior;
    const float3 in_direction = si->direction;

    si->albedo = make_float3(color);
    float cos_theta = dot(in_direction, si->n);
    bool into = cos_theta < 0;
    const float3 outward_normal = into ? si->n : -si->n;
    const float refraction_ratio = into ? (1.0 / ior) : ior;

    float3 unit_direction = normalize(in_direction);
    cos_theta = fabs(cos_theta);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    unsigned int seed = si->seed;
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    if (cannot_refract || rnd(seed) < fresnel(cos_theta, refraction_ratio))
        scattered = reflect(unit_direction, si->n);
    else
        scattered = refract(unit_direction, outward_normal, refraction_ratio);
    si->trace_terminate = false;
    si->emission = make_float3(0.0f);
    si->seed = seed;
}

extern "C" __device__ void __direct_callable__metal(SurfaceInfo* si, void* material_data, float3& scattered)
{
    const MetalData* metal = (MetalData*)material_data;
    // Direct callableプログラムによって、テクスチャ色を取得
    const float4 color = optixDirectCall<float4, SurfaceInfo*, void*>(
        metal->texture_prg_id, si, metal->texture_data
        );

    unsigned int seed = si->seed;
    scattered = reflect(si->direction, si->n) + metal->fuzz * randomInUnitSphere(seed);
    si->albedo = make_float3(color);
    si->trace_terminate = false;
    si->emission = make_float3(0.0f);
    si->seed = seed;
}

extern "C" __device__ float4 __direct_callable__constant(SurfaceInfo* /* si */ , void* texture_data)
{
    const ConstantData* constant = (ConstantData*)texture_data;
    return constant->color;
}

extern "C" __device__ float4 __direct_callable__checker(SurfaceInfo* si, void* texture_data)
{
    const CheckerData* checker = (CheckerData*)texture_data;
    const bool is_odd = sinf(si->texcoord.x * M_PIf * checker->scale) * sinf(si->texcoord.y * M_PIf * checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}