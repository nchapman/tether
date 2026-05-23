// Mitchell-Netravali separable bicubic resampling in linear-light, with
// scale-aware tap count and a 2× box-filter mip prefilter for heavy
// downscale.
//
// Three entry points:
//   - `mip_box_down`: 2× linear-light box downsample. Invoked
//     repeatedly by the host until the next Mitchell pass is within
//     2× scale.
//   - `horizontal`: Mitchell along x. Reads sRGB-encoded source,
//     writes linear-light Rgba16Float intermediate.
//   - `vertical`: Mitchell along y. Reads linear intermediate, writes
//     sRGB-encoded Rgba8Unorm output.
//
// Conventions that need to match the Rust reference exactly (see
// `crates/tether-scaler/src/reference.rs`):
//   - Texel-center mapping: `center = (out + 0.5) * scale - 0.5`.
//   - Half-tap offset: `i0 = floor(center) - n_taps/2 + 1`.
//   - Edge clamp: `clamp(x, 0, dim-1)` (extend-by-replicate).
//   - Mitchell B = C = 1/3 (compile-time const — change only with a
//     matching change in reference.rs and the verification baseline).
//
// Output format is Rgba8Unorm, not Bgra8Unorm. Bgra8Unorm storage
// textures require the `BGRA8UNORM_STORAGE` wgpu feature which is not
// portable across backends; Rgba8Unorm storage is core. Downstream
// chroma shaders read .rgb regardless of source format.
//
// **Don't add workgroup-shared-memory tiling for the Mitchell passes.**
// Measured and reverted: on Intel iGPU (the production target floor)
// the tiled version was 1.5–2.8× *slower* across every realistic
// (src, dst) pair. The naive `textureLoad` loop already hits the
// hardware L1/L2 texture cache hard — 64 threads in an 8×8 workgroup
// produce a tight access cluster the cache exploits in hardware.
// Adding a workgroupBarrier + manual cache layer pays synchronization
// cost without bandwidth savings. The theoretical reuse argument
// (`8 * n_taps` reads collapsed to `8 + n_taps - 1`) assumes the
// underlying texture cache is doing nothing; it isn't. Don't re-add
// without an updated benchmark.

const B: f32 = 1.0 / 3.0;
const C: f32 = 1.0 / 3.0;

fn mitchell(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1.0) {
        return ((12.0 - 9.0 * B - 6.0 * C) * ax * ax * ax
              + (-18.0 + 12.0 * B + 6.0 * C) * ax * ax
              + (6.0 - 2.0 * B)) / 6.0;
    } else if (ax < 2.0) {
        return ((-B - 6.0 * C) * ax * ax * ax
              + (6.0 * B + 30.0 * C) * ax * ax
              + (-12.0 * B - 48.0 * C) * ax
              + (8.0 * B + 24.0 * C)) / 6.0;
    }
    return 0.0;
}

// IEC 61966-2-1 sRGB transfer — piecewise, not the gamma-2.2 approx.
// The approximation is wrong enough at low values that it would visibly
// dim text antialiasing under downscale; the cost of the accurate form
// is negligible vs the Mitchell weight evaluations.
fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}
fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) { return 12.92 * c; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn decode_rgb(srgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(srgb.r), srgb_to_linear(srgb.g), srgb_to_linear(srgb.b));
}
fn encode_rgb(lin: vec3<f32>) -> vec3<f32> {
    let l = clamp(lin, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec3<f32>(linear_to_srgb(l.r), linear_to_srgb(l.g), linear_to_srgb(l.b));
}

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    n_taps: u32,
    // Sample-position offset (in source-pixel units) added to the
    // computed kernel center. For Y planes and the legacy RGB paths,
    // this is zero. For UV planes in NV12-with-left-cosited-chroma,
    // the host passes `-(scale - 1) * 0.5` on the appropriate axis so
    // the dst chroma grid lines up with the dst luma grid — without
    // it, naive plane-independent Mitchell shifts UV by half a luma
    // pixel and text edges fringe colour.
    chroma_offset_x: f32,
    chroma_offset_y: f32,
    _pad: u32,
};

// === Horizontal pass ===
@group(0) @binding(0) var src_h: texture_2d<f32>;
@group(0) @binding(1) var dst_h: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params_h: Params;

@compute @workgroup_size(8, 8, 1)
fn horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_h.dst_w || gid.y >= params_h.src_h) {
        return;
    }
    let scale = f32(params_h.src_w) / f32(params_h.dst_w);
    let support = max(scale, 1.0);
    let center = (f32(gid.x) + 0.5) * scale - 0.5 + params_h.chroma_offset_x;
    let half_taps = i32(params_h.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_h.n_taps; k = k + 1u) {
        let x = i0 + i32(k);
        let xc = clamp(x, 0, i32(params_h.src_w) - 1);
        let w = mitchell((f32(x) - center) / support);
        let srgb = textureLoad(src_h, vec2<i32>(xc, i32(gid.y)), 0).rgb;
        sum = sum + decode_rgb(srgb) * w;
        weight_sum = weight_sum + w;
    }
    // Mitchell weights sum analytically to ~1 over their full support,
    // but with clamped edges + finite tap counts the sum can drift.
    // Normalize. Guard the divide because at degenerate edge cases the
    // sum can approach zero, which would amplify noise to garbage.
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    textureStore(dst_h, vec2<i32>(gid.xy), vec4<f32>(sum / ws, 1.0));
}

// === Vertical pass ===
@group(0) @binding(0) var src_v: texture_2d<f32>;
@group(0) @binding(1) var dst_v: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params_v: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v.dst_w || gid.y >= params_v.dst_h) {
        return;
    }
    let scale = f32(params_v.src_h) / f32(params_v.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v.chroma_offset_y;
    let half_taps = i32(params_v.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        let lin = textureLoad(src_v, vec2<i32>(i32(gid.x), yc), 0).rgb;
        sum = sum + lin * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    let lin = sum / ws;
    textureStore(dst_v, vec2<i32>(gid.xy), vec4<f32>(encode_rgb(lin), 1.0));
}

// === Linear-light variant ===
//
// Same algorithm but input/output are already in linear light — no
// sRGB transfer either way. Used by the client renderer, where the
// YUV→RGB fragment shader emits linear-light values into an
// `Rgba16Float` intermediate; Mitchell-upscaling those values
// directly keeps the math correct (the post-Mitchell swapchain blit
// applies the surface's OETF on write, same as the original path).

@group(0) @binding(0) var src_h_lin: texture_2d<f32>;
@group(0) @binding(1) var dst_h_lin: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params_h_lin: Params;

@compute @workgroup_size(8, 8, 1)
fn horizontal_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_h_lin.dst_w || gid.y >= params_h_lin.src_h) {
        return;
    }
    let scale = f32(params_h_lin.src_w) / f32(params_h_lin.dst_w);
    let support = max(scale, 1.0);
    let center = (f32(gid.x) + 0.5) * scale - 0.5 + params_h_lin.chroma_offset_x;
    let half_taps = i32(params_h_lin.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_h_lin.n_taps; k = k + 1u) {
        let x = i0 + i32(k);
        let xc = clamp(x, 0, i32(params_h_lin.src_w) - 1);
        let w = mitchell((f32(x) - center) / support);
        sum = sum + textureLoad(src_h_lin, vec2<i32>(xc, i32(gid.y)), 0).rgb * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    textureStore(dst_h_lin, vec2<i32>(gid.xy), vec4<f32>(sum / ws, 1.0));
}

@group(0) @binding(0) var src_v_lin: texture_2d<f32>;
@group(0) @binding(1) var dst_v_lin: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params_v_lin: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v_lin.dst_w || gid.y >= params_v_lin.dst_h) {
        return;
    }
    let scale = f32(params_v_lin.src_h) / f32(params_v_lin.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v_lin.chroma_offset_y;
    let half_taps = i32(params_v_lin.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v_lin.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v_lin.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        sum = sum + textureLoad(src_v_lin, vec2<i32>(i32(gid.x), yc), 0).rgb * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    textureStore(dst_v_lin, vec2<i32>(gid.xy), vec4<f32>(sum / ws, 1.0));
}

// === YUV plane vertical passes ===
//
// The horizontal pass for Y and UV planes reuses `horizontal_linear`:
// `texture_2d<f32>` sampling on an R8Unorm source returns (r, 0, 0)
// and on Rg8Unorm returns (r, g, 0), and the intermediate is the same
// Rgba16Float used by every other path. Only the *vertical* pass
// differs because the destination format is scalar R8Unorm (Y plane)
// or two-channel Rg8Unorm (UV plane).
//
// Both entry points read the same Rgba16Float intermediate (samples
// 3 channels but only the first 1 or 2 are meaningful) and store the
// relevant channels back. No transfer function: NV12 planes are
// already in the encoded domain VT expects (limited-range BT.709 Y or
// UV); Mitchell on gamma-encoded Y is mathematically imperfect but
// visually fine at 8-bit and is the project trade-off (see plan).

@group(0) @binding(0) var src_v_plane_r: texture_2d<f32>;
@group(0) @binding(1) var dst_v_plane_r: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var<uniform> params_v_plane_r: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical_plane_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v_plane_r.dst_w || gid.y >= params_v_plane_r.dst_h) {
        return;
    }
    let scale = f32(params_v_plane_r.src_h) / f32(params_v_plane_r.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v_plane_r.chroma_offset_y;
    let half_taps = i32(params_v_plane_r.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v_plane_r.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v_plane_r.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        sum = sum + textureLoad(src_v_plane_r, vec2<i32>(i32(gid.x), yc), 0).r * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    let v = clamp(sum / ws, 0.0, 1.0);
    textureStore(dst_v_plane_r, vec2<i32>(gid.xy), vec4<f32>(v, 0.0, 0.0, 1.0));
}

@group(0) @binding(0) var src_v_plane_rg: texture_2d<f32>;
@group(0) @binding(1) var dst_v_plane_rg: texture_storage_2d<rg8unorm, write>;
@group(0) @binding(2) var<uniform> params_v_plane_rg: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical_plane_rg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v_plane_rg.dst_w || gid.y >= params_v_plane_rg.dst_h) {
        return;
    }
    let scale = f32(params_v_plane_rg.src_h) / f32(params_v_plane_rg.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v_plane_rg.chroma_offset_y;
    let half_taps = i32(params_v_plane_rg.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec2<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v_plane_rg.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v_plane_rg.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        sum = sum + textureLoad(src_v_plane_rg, vec2<i32>(i32(gid.x), yc), 0).rg * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    let v = clamp(sum / ws, vec2<f32>(0.0), vec2<f32>(1.0));
    textureStore(dst_v_plane_rg, vec2<i32>(gid.xy), vec4<f32>(v.x, v.y, 0.0, 1.0));
}

// === Mip prefilter (2× box downsample, linear-light) ===
// Invoked when the desired downscale ratio exceeds 2×. Produces an
// sRGB-encoded intermediate (same shape as the original capture) so
// successive mip levels or the Mitchell pass read sRGB without
// special-casing the source.
@group(0) @binding(0) var mip_src: texture_2d<f32>;
@group(0) @binding(1) var mip_dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn mip_box_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(mip_dst);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }
    let sx = i32(gid.x * 2u);
    let sy = i32(gid.y * 2u);
    let src_dims = textureDimensions(mip_src);
    let sx1 = min(sx + 1, i32(src_dims.x) - 1);
    let sy1 = min(sy + 1, i32(src_dims.y) - 1);
    let p00 = decode_rgb(textureLoad(mip_src, vec2<i32>(sx,  sy ), 0).rgb);
    let p10 = decode_rgb(textureLoad(mip_src, vec2<i32>(sx1, sy ), 0).rgb);
    let p01 = decode_rgb(textureLoad(mip_src, vec2<i32>(sx,  sy1), 0).rgb);
    let p11 = decode_rgb(textureLoad(mip_src, vec2<i32>(sx1, sy1), 0).rgb);
    let lin = (p00 + p10 + p01 + p11) * 0.25;
    textureStore(mip_dst, vec2<i32>(gid.xy), vec4<f32>(encode_rgb(lin), 1.0));
}
