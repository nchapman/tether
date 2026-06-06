// Planar YUV 4:4:4 fragment shader.
//
// NVIDIA NVDEC exports HEVC Main 4:4:4 8-bit as DRM_FORMAT_YUV444 (`YU24`):
// three full-resolution R8 planes. This shader is the planar sibling of
// shader_yuv444.wgsl; only the sample step differs.

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

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var s: sampler;

@group(2) @binding(0) var<uniform> color_params: vec4<u32>;
const TRANSFER_KIND_SRGB: u32 = 1u;

fn limited_y_to_normalized(y_lim: f32) -> f32 {
    return (y_lim - 16.0 / 255.0) * (255.0 / 219.0);
}

fn limited_c_to_normalized(c_lim: f32) -> f32 {
    return (c_lim - 128.0 / 255.0) * (255.0 / 224.0);
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
    let y_lim = textureSample(y_tex, s, in.uv).r;
    let u_lim = textureSample(u_tex, s, in.uv).r;
    let v_lim = textureSample(v_tex, s, in.uv).r;

    let y = limited_y_to_normalized(y_lim);
    let u = limited_c_to_normalized(u_lim);
    let v = limited_c_to_normalized(v_lim);

    let rgb_gamma = bt709_ycbcr_to_rgb_gamma(y, u, v);
    let rgb_linear = apply_eotf(rgb_gamma);
    return vec4<f32>(rgb_linear, 1.0);
}
