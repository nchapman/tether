struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// .xy = (x_scale, y_scale) applied to NDC positions for aspect-ratio
// preserving letterbox / pillarbox. .zw padded out to a vec4 because
// uniform buffers in WGSL require 16-byte alignment.
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

// NV12: full-res single-channel Y plus half-res two-channel UV with U
// in .r and V in .g. One sample of the UV texture yields both chroma
// components, replacing the older path's pair of single-channel taps.
// Bilinear filter on the chroma sampler is free on every GPU we care
// about and the visual difference vs. nearest is invisible at 4:2:0.
@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

// Color params. .x is the EOTF dispatch tag (see Rust constants
// TRANSFER_KIND_* in gpu/mod.rs); .yzw are reserved padding for
// future axes (matrix kind / range kind when they grow shader
// variants).
@group(2) @binding(0) var<uniform> color_params: vec4<u32>;
const TRANSFER_KIND_BT709: u32 = 0u;
const TRANSFER_KIND_SRGB: u32 = 1u;

// =============================================================
// YUV -> display: three independent transforms, each swappable.
// =============================================================
//
// The pipeline has to track three things that vary independently
// across codecs / displays. Splitting the shader into one function
// per concern means a future HDR or BT.2020 path swaps exactly the
// function that changes — no copy-paste of the whole fragment body.
//
//   1. RANGE
//      How the integer pixel values map to normalized luma/chroma.
//      Limited (16..235 / 16..240) for broadcast and what every
//      Tether encoder produces today; full (0..255) for capture
//      sources that hand us PC-range data.
//
//   2. MATRIX (a.k.a. color-conversion matrix)
//      The 3x3 that turns normalized Y'CbCr into *gamma-encoded*
//      R'G'B'. BT.709 today (sRGB-gamut content); BT.601 for
//      legacy SD; BT.2020 for HDR. The R'G'B' output of this step
//      is in the SAME GAMMA SPACE as the source content — it is
//      NOT linear light. (The apostrophe in R'G'B' is load-
//      bearing; missing this is the single most common cause of
//      "the video looks washed out" bugs.)
//
//   3. TRANSFER FUNCTION (EOTF)
//      The 1D curve that lifts gamma-encoded R'G'B' to linear
//      light. BT.709 EOTF for SDR broadcast (what we emit today);
//      sRGB EOTF for desktop-captured content; PQ (BT.2100) and
//      HLG for HDR. Linear light is the universal interchange —
//      blending, gamut conversion, and tone mapping all want it.
//
// We output **linear** light to the fragment surface. wgpu applies
// the surface format's OETF on write (sRGB encoding for an
// `Bgra8UnormSrgb` surface) so the bytes that hit the display are
// in the display's expected encoding. The HDR path will swap the
// surface format (e.g. `Rgba16Float` with PQ/HLG signaling) and
// the linear-light output stays the same.
//
// Stream-side color metadata (range / matrix / transfer triplet)
// isn't carried on the wire yet; until it is, this shader is
// hard-pinned to BT.709 limited-range — what every encoder we
// configure today produces.

fn limited_y_to_normalized(y_lim: f32) -> f32 {
    // Y[16..235] -> [0..1]. Out-of-range values stay out-of-range;
    // we don't clamp because the matrix and EOTF below tolerate it
    // (and clamping here would crush legitimate overshoots from
    // chroma siting / encoder rate-control).
    return (y_lim - 16.0 / 255.0) * (255.0 / 219.0);
}

fn limited_c_to_normalized(c_lim: f32) -> f32 {
    // Cb/Cr[16..240] -> [-0.5..0.5].
    return (c_lim - 128.0 / 255.0) * (255.0 / 224.0);
}

fn bt709_ycbcr_to_rgb_gamma(y: f32, u: f32, v: f32) -> vec3<f32> {
    // BT.709 inverse matrix. Output is gamma-encoded R'G'B' in the
    // [0,1] nominal range (possibly with small out-of-range
    // excursions from limited-range overshoots).
    let r = y + 1.5748 * v;
    let g = y - 0.1873 * u - 0.4681 * v;
    let b = y + 1.8556 * u;
    return vec3<f32>(r, g, b);
}

fn bt709_eotf_component(v: f32) -> f32 {
    // BT.709 EOTF (Rec. ITU-R BT.709-6, eq. 1.2 inverted). Piecewise
    // linear-then-power with the same break point and gamma as the
    // BT.709 OETF used by capture/encode hardware. Clamp negatives
    // to zero before `pow` to handle the tiny excursions limited-
    // range chroma can produce after the matrix; without this, the
    // pow on a negative value is undefined behaviour in WGSL.
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
    // sRGB EOTF (IEC 61966-2-1). The principled transfer for desktop
    // capture sources whose framebuffer bytes are sRGB-encoded
    // (which is to say: macOS / Wayland / Windows compositor
    // output). Same shape as the BT.709 EOTF but a different
    // break-point and exponent — the curves agree at the endpoints
    // and diverge by up to ~5% in midtones.
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
    // Per-frame uniform branch — every fragment in a draw takes the
    // same path, so there's no warp divergence cost. Modern GPUs
    // optimise this to a uniform control flow predicate.
    if (color_params.x == TRANSFER_KIND_SRGB) {
        return srgb_eotf(rgb_gamma);
    }
    return bt709_eotf(rgb_gamma);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // 1. SAMPLE
    let y_lim = textureSample(y_tex, s, in.uv).r;
    let chroma_lim = textureSample(uv_tex, s, in.uv).rg;

    // 2. RANGE: limited -> normalized.
    let y = limited_y_to_normalized(y_lim);
    let u = limited_c_to_normalized(chroma_lim.r);
    let v = limited_c_to_normalized(chroma_lim.g);

    // 3. MATRIX: BT.709 Y'CbCr -> gamma-encoded R'G'B'.
    let rgb_gamma = bt709_ycbcr_to_rgb_gamma(y, u, v);

    // 4. TRANSFER: EOTF -> linear light. `apply_eotf` dispatches on
    // the negotiated `color_params.x` so a stream that advertised
    // sRGB transfer (desktop capture) doesn't get the BT.709 EOTF
    // applied to it — that mismatch is the ≤~5% midtone lift the
    // tolerance-18 regression test currently tolerates.
    let rgb_linear = apply_eotf(rgb_gamma);

    // wgpu applies the surface format's OETF on write. Output is
    // therefore linear; the display receives correctly-encoded
    // pixels regardless of the surface format we picked.
    return vec4<f32>(rgb_linear, 1.0);
}

// HEVC Main444 sessions use a separate shader module (shader_yuv444.wgsl).
// Two distinct module-scope binding declarations at @group(0) @binding(0)
// in one WGSL file is a validation error per spec — even when only one
// entry point is compiled into a given pipeline. Keep them in separate
// files so each module's bindings stand on their own.
