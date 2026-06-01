// YUV 4:4:4 10-bit packed (Y410) fragment shader.
//
// 10-bit sibling of `shader_yuv444.wgsl`. Linux VAAPI decodes HEVC Main
// 4:4:4 10-bit to packed Y410 (VA_FOURCC_Y410 / DRM_FORMAT_Y410, imported
// as VK_FORMAT_A2B10G10R10_UNORM_PACK32 / wgpu `Rgb10a2Unorm`): one
// 32-bit 10:10:10:2 word per pixel.
//
//   Channel mapping (EMPIRICALLY VERIFIED, not the DRM-standard reading):
//     .r = Y, .g = U, .b = V, .a = don't-care
//
//   Intel's media-driver emits Y410 with the SAME Rgb10a2 channel→
//   component layout it consumes as XV30 input (see the encode-side
//   `bgra_to_xv30.wgsl`, which packs R=Y/G=U/B=V) — *not* the
//   DRM_FORMAT_Y410 spec order ("A:Cr:Y:Cb" → R=Cb). Reading it the
//   spec way scrambles the channels: the `roundtrip_hevc_main444_10bit_
//   identity` hardware cell drops to SSIM≈0.75 / 9.6 dB on the spec
//   order and passes at SSIM≈0.998 / 61.7 dB on this one.
//
// Unlike P010/P410 (10-bit MSB-aligned in a 16-bit `R16Unorm` cell, which
// the sampler reads back as `value*64/65535`), `Rgb10a2Unorm` is a native
// 10-bit format: the sampler returns `code/1023` directly. So the
// limited-range breakpoints here use `/1023` denominators — distinct from
// the `/65535` MSB-aligned constants in `shader.wgsl`'s 10-bit branch.
//
// Matrix + EOTF are identical to the other YUV shaders; keep them in sync.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(1) @binding(0) var<uniform> scale: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
        vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );
    var out: VsOut;
    out.pos = vec4(positions[vi] * scale.xy, 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@group(0) @binding(0) var packed_tex: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@group(2) @binding(0) var<uniform> color_params: vec4<u32>;
const TRANSFER_KIND_BT709: u32 = 0u;
const TRANSFER_KIND_SRGB: u32 = 1u;

// Native 10-bit limited-range (no MSB-align): black=64/1023,
// white=940/1023 (range 876), chroma neutral=512/1023 (range 896).
// Exactly inverts `bgra_to_xv30.wgsl`'s Y_OFFSET_10/Y_SCALE_10 /
// UV_OFFSET_10/UV_SCALE_10.
fn limited_y_to_normalized(y_lim: f32) -> f32 {
    return (y_lim - 64.0 / 1023.0) * (1023.0 / 876.0);
}

fn limited_c_to_normalized(c_lim: f32) -> f32 {
    return (c_lim - 512.0 / 1023.0) * (1023.0 / 896.0);
}

fn bt709_ycbcr_to_rgb_gamma(y: f32, u: f32, v: f32) -> vec3<f32> {
    let r = y + 1.5748 * v;
    let g = y - 0.1873 * u - 0.4681 * v;
    let b = y + 1.8556 * u;
    return vec3<f32>(r, g, b);
}

fn bt709_eotf_component(v: f32) -> f32 {
    let vc = max(v, 0.0);
    if (vc < 0.081) {
        return vc / 4.5;
    }
    return pow((vc + 0.099) / 1.099, 1.0 / 0.45);
}

fn bt709_eotf(rgb_gamma: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        bt709_eotf_component(rgb_gamma.x),
        bt709_eotf_component(rgb_gamma.y),
        bt709_eotf_component(rgb_gamma.z),
    );
}

fn srgb_eotf_component(v: f32) -> f32 {
    let vc = max(v, 0.0);
    if (vc <= 0.04045) {
        return vc / 12.92;
    }
    return pow((vc + 0.055) / 1.055, 2.4);
}

fn srgb_eotf(rgb_gamma: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_eotf_component(rgb_gamma.x),
        srgb_eotf_component(rgb_gamma.y),
        srgb_eotf_component(rgb_gamma.z),
    );
}

fn apply_eotf(rgb_gamma: vec3<f32>) -> vec3<f32> {
    if (color_params.x == TRANSFER_KIND_SRGB) {
        return srgb_eotf(rgb_gamma);
    }
    return bt709_eotf(rgb_gamma);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Channel mapping mirrors the encoder side (`bgra_to_xv30.wgsl`):
    // .r = Y, .g = U, .b = V. Intel's VAAPI decoder emits Y410 with the
    // same Rgb10a2 channel→component layout it consumed as XV30, so the
    // import unpacks exactly what the encode pass packed. (Pinned by the
    // `roundtrip_hevc_main444_10bit_identity` hardware cell.)
    let p = textureSample(packed_tex, s, in.uv);
    let y = limited_y_to_normalized(p.r);
    let u = limited_c_to_normalized(p.g);
    let v = limited_c_to_normalized(p.b);

    let rgb_gamma = bt709_ycbcr_to_rgb_gamma(y, u, v);
    let rgb_linear = apply_eotf(rgb_gamma);
    return vec4<f32>(rgb_linear, 1.0);
}
