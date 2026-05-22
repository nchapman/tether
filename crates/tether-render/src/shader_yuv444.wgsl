// YUV 4:4:4 packed fragment shader.
//
// Sibling of `shader.wgsl` for sessions that negotiate HEVC Main444.
// Decoded surfaces arrive as packed XYUV (DRM_FORMAT_XYUV8888 /
// VA_FOURCC_XYUV): one Rgba8Unorm-shaped texture, 32 bpp, memory
// byte order [V, U, Y, X]. Reading via wgpu's R8G8B8A8_UNORM mapping:
//
//   .r = byte 0 = V
//   .g = byte 1 = U
//   .b = byte 2 = Y
//   .a = byte 3 = X (don't-care)
//
// Why packed (not planar): ffmpeg's `vaapi_drm_format_map` has no
// entry for planar YUV444P over DRM_PRIME — `vaExportSurfaceHandle`
// on a 4:4:4 surface in production returns the packed XYUV shape
// regardless. The encoder side feeds the same packed format via
// gpuconvert's BGRA→XYUV compute pass; see
// `crates/tether-gpuconvert/src/dmabuf_export/shared_yuv444.rs`.
//
// The Range / Matrix / EOTF pipeline is identical to the NV12 sibling;
// only the SAMPLE step changes. Keep the matrix and EOTF constants in
// sync between the two shaders — the encoder's VUI is shared, so a
// divergence here produces a colour shift visible only on 4:4:4
// sessions and not on the 4:2:0 baseline used as a regression check.
//
// Each shader compiles into its own pipeline against its own bind-group
// layout (see `gpu/mod.rs::new` for the chroma-keyed dispatch). Keeping
// them in separate files means each WGSL module's module-scope binding
// declarations stand on their own — sharing a file would force two
// distinct variables at `@group(0) @binding(0)`, which is a WGSL
// validation error even when only one entry point is selected at
// pipeline build.

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

// Limited-range BT.709 -> normalised. 8-bit only by construction:
// `render_layout_for` routes 10-bit 4:4:4 to Biplanar16 (P410), so a
// PackedXYUV pipeline can only be built for 8-bit data. If that
// invariant ever changes, mirror `shader.wgsl`'s RANGE_KIND_LIMITED_10
// branch here too.
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
    // XYUV8888 byte order V, U, Y, X → Rgba8 .rgba.
    let p = textureSample(packed_tex, s, in.uv);
    let y_lim = p.b;
    let u_lim = p.g;
    let v_lim = p.r;

    let y = limited_y_to_normalized(y_lim);
    let u = limited_c_to_normalized(u_lim);
    let v = limited_c_to_normalized(v_lim);

    let rgb_gamma = bt709_ycbcr_to_rgb_gamma(y, u, v);
    let rgb_linear = apply_eotf(rgb_gamma);
    return vec4<f32>(rgb_linear, 1.0);
}
