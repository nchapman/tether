// Native D3D11 YUV→RGB shader for the Windows client renderer.
//
// Line-for-line counterpart to `shader.wgsl` (the wgpu/Linux path): the
// same three swappable transforms — range expand → BT.709 matrix →
// EOTF — and the same letterbox-scaled fullscreen quad. Keeping the math
// identical is what lets a frame look the same on either backend.
//
// NV12 8-bit or P010 10-bit, limited-range (Windows is 4:2:0 today). The
// Y plane is an R8/R16 SRV (t0), the UV plane an R8G8/R16G16 SRV (t1) —
// the SRV format follows the bit depth — both opened from the decoder's
// shared NT handles. Output is **linear light**; the render
// target is a `*_SRGB` view so the hardware applies the sRGB OETF on
// store — the exact parity with wgpu's sRGB surface write.

cbuffer Params : register(b0)
{
    // .xy = (x_scale, y_scale) applied to NDC for aspect-preserving
    // letterbox / pillarbox; .zw padding (cbuffers are 16-byte aligned).
    float4 scale;
    // .x = transfer (EOTF) kind, .y = range kind. Mirrors the
    // `color_params` uniform in shader.wgsl.
    uint4 color_params;
};

static const uint TRANSFER_KIND_SRGB = 1u;
static const uint RANGE_KIND_LIMITED_10 = 1u;

Texture2D<float> y_tex : register(t0);
Texture2D<float2> uv_tex : register(t1);
SamplerState samp : register(s0);

struct VsOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VsOut vs(uint vid : SV_VertexID)
{
    // Two-triangle fullscreen quad. Same position/uv pairing as
    // shader.wgsl: D3D and wgpu agree on clip-space-y-up and
    // texture-v-down, so the mapping carries over unchanged.
    const float2 positions[6] =
    {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
    };
    const float2 uvs[6] =
    {
        float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0),
        float2(0.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0)
    };
    VsOut o;
    o.pos = float4(positions[vid] * scale.xy, 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

float limited_y_to_normalized(float y_lim)
{
    if (color_params.y == RANGE_KIND_LIMITED_10)
        return (y_lim - 4096.0 / 65535.0) * (65535.0 / 56064.0);
    return (y_lim - 16.0 / 255.0) * (255.0 / 219.0);
}

float limited_c_to_normalized(float c_lim)
{
    if (color_params.y == RANGE_KIND_LIMITED_10)
        return (c_lim - 32768.0 / 65535.0) * (65535.0 / 57344.0);
    return (c_lim - 128.0 / 255.0) * (255.0 / 224.0);
}

float3 bt709_ycbcr_to_rgb_gamma(float y, float u, float v)
{
    // BT.709 inverse matrix → gamma-encoded R'G'B'.
    float r = y + 1.5748 * v;
    float g = y - 0.1873 * u - 0.4681 * v;
    float b = y + 1.8556 * u;
    return float3(r, g, b);
}

float bt709_eotf_component(float v)
{
    // Clamp negatives before pow (limited-range chroma can overshoot).
    float vc = max(v, 0.0);
    if (vc < 0.081)
        return vc / 4.5;
    return pow((vc + 0.099) / 1.099, 1.0 / 0.45);
}

float srgb_eotf_component(float v)
{
    float vc = max(v, 0.0);
    if (vc <= 0.04045)
        return vc / 12.92;
    return pow((vc + 0.055) / 1.055, 2.4);
}

float3 apply_eotf(float3 rgb_gamma)
{
    if (color_params.x == TRANSFER_KIND_SRGB)
        return float3(
            srgb_eotf_component(rgb_gamma.x),
            srgb_eotf_component(rgb_gamma.y),
            srgb_eotf_component(rgb_gamma.z));
    return float3(
        bt709_eotf_component(rgb_gamma.x),
        bt709_eotf_component(rgb_gamma.y),
        bt709_eotf_component(rgb_gamma.z));
}

float4 ps(VsOut input) : SV_Target
{
    // 1. SAMPLE (bilinear). Y in R8, UV interleaved in R8G8.
    float y_lim = y_tex.Sample(samp, input.uv).r;
    float2 chroma_lim = uv_tex.Sample(samp, input.uv).rg;

    // 2. RANGE: limited → normalized.
    float y = limited_y_to_normalized(y_lim);
    float u = limited_c_to_normalized(chroma_lim.r);
    float v = limited_c_to_normalized(chroma_lim.g);

    // 3. MATRIX: BT.709 Y'CbCr → gamma R'G'B'.
    float3 rgb_gamma = bt709_ycbcr_to_rgb_gamma(y, u, v);

    // 4. TRANSFER: EOTF → linear light. The _SRGB RTV re-encodes on write.
    float3 rgb_linear = apply_eotf(rgb_gamma);
    return float4(rgb_linear, 1.0);
}
