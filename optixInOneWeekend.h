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

struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int max_depth;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;

    // 動的にシーンを変更するための時間とマウス位置
    // ホスト側(CPU)で毎フレーム値を更新して、デバイス側(GPU)に転送する
    float time;
    float mouse_x;
    float mouse_y;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};

struct SphereData
{
    // 球の中心
    float3 center;
    // 球の半径
    float radius;
};

struct MeshData
{
    // メッシュの頂点
    float3* vertices;
    // 三角形を構成するための頂点番号3点
    uint3* indices;
};

struct LambertianData {
    // Lambert マテリアルの色
    void* texture_data;
    unsigned int texture_prg_id;
};

struct DielectricData {
    // 誘電体の色
    void* texture_data;
    unsigned int texture_prg_id;
    // 屈折率
    float ior; 
};

struct MetalData {
    // 金属の色
    void* texture_data;
    unsigned int texture_prg_id;
    // 金属の疑似粗さを指定するパラメータ
    float fuzz;
};

struct Material {
    void* data; 
    unsigned int prg_id;
};

struct ConstantData
{
    float4 color;
};

struct CheckerData
{
    float4 color1; 
    float4 color2;
    float scale;
};

struct HitGroupData
{
    // 物体形状に関するデータ
    // デバイス上に確保されたポインタを紐づける
    // 共用体を使わずに汎用ポインタにすることで、
    // 異なるデータ型の構造体を追加したいときに対応しやすくなる。
    void* shape_data;

    //// マテリアル(Lambertian, Glass, Metal)のデータ
    //// デバイス上に確保されたポインタを紐づけておく
    //// 物体形状同様に汎用ポインタを使う
    //void* material_data;

    //// マテリアルにおける散乱方向や色を計算するためのCallablesプログラムのID
    //// OptiX 7.x では仮想関数が使えないので、Callablesプログラムを使って
    //// 疑似的なポリモーフィズムを実現する
    //unsigned int material_prg_id;

    Material material;
};

struct EmptyData
{

};