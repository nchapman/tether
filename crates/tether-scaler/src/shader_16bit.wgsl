// 10-bit YUV plane vertical passes (R16Unorm / Rg16Unorm storage).
//
// Companion module to `shader.wgsl`. Lives in a separate WGSL file
// because wgpu/naga records `STORAGE_TEXTURE_16BIT_NORM_FORMATS` as a
// module-level required capability (not per-entry-point), so an 8-bit
// session running on an adapter that lacks the 16-bit-norm storage
// feature would fail shader-module compilation if these entry points
// were in `shader.wgsl`. Loaded only by
// `Pipelines::build_with_plane_storage_16bit`.
//
// 10-bit NV12 (`'x420'` / `'xf20'` / `'P010'`) stores 10 bits MSB-
// aligned in 16-bit cells; `texture_storage_2d<r16unorm, write>`
// writes the full 16 bits, so the bottom 6 bits land as zero — VT
// and the decoder side ignore them per the chroma-loc-type=0
// convention, same as the renderer's biplanar 10-bit fragment
// shader's `luma_scale` compensation on read.
//
// The Mitchell-Netravali math is identical to the R8/Rg8 variants
// in `shader.wgsl`. The horizontal pass for both bit depths reuses
// `horizontal_linear` from `shader.wgsl` (texture_2d<f32> sampling
// returns values normalised to [0, 1] for R16Unorm sources, same
// interface as R8). Only the destination storage format differs.

// --- shared Mitchell math + Params struct (duplicated from
// shader.wgsl because WGSL has no #include).

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

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    n_taps: u32,
    chroma_offset_x: f32,
    chroma_offset_y: f32,
    _pad: u32,
};

// === Y plane vertical (R16Unorm storage) ===
@group(0) @binding(0) var src_v_plane_r16: texture_2d<f32>;
@group(0) @binding(1) var dst_v_plane_r16: texture_storage_2d<r16unorm, write>;
@group(0) @binding(2) var<uniform> params_v_plane_r16: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical_plane_r16(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v_plane_r16.dst_w || gid.y >= params_v_plane_r16.dst_h) {
        return;
    }
    let scale = f32(params_v_plane_r16.src_h) / f32(params_v_plane_r16.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v_plane_r16.chroma_offset_y;
    let half_taps = i32(params_v_plane_r16.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v_plane_r16.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v_plane_r16.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        sum = sum + textureLoad(src_v_plane_r16, vec2<i32>(i32(gid.x), yc), 0).r * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    let v = clamp(sum / ws, 0.0, 1.0);
    textureStore(dst_v_plane_r16, vec2<i32>(gid.xy), vec4<f32>(v, 0.0, 0.0, 1.0));
}

// === UV plane vertical (Rg16Unorm storage) ===
@group(0) @binding(0) var src_v_plane_rg16: texture_2d<f32>;
@group(0) @binding(1) var dst_v_plane_rg16: texture_storage_2d<rg16unorm, write>;
@group(0) @binding(2) var<uniform> params_v_plane_rg16: Params;

@compute @workgroup_size(8, 8, 1)
fn vertical_plane_rg16(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params_v_plane_rg16.dst_w || gid.y >= params_v_plane_rg16.dst_h) {
        return;
    }
    let scale = f32(params_v_plane_rg16.src_h) / f32(params_v_plane_rg16.dst_h);
    let support = max(scale, 1.0);
    let center = (f32(gid.y) + 0.5) * scale - 0.5 + params_v_plane_rg16.chroma_offset_y;
    let half_taps = i32(params_v_plane_rg16.n_taps / 2u);
    let i0 = i32(floor(center)) - half_taps + 1;
    var sum = vec2<f32>(0.0);
    var weight_sum = 0.0;
    for (var k: u32 = 0u; k < params_v_plane_rg16.n_taps; k = k + 1u) {
        let y = i0 + i32(k);
        let yc = clamp(y, 0, i32(params_v_plane_rg16.src_h) - 1);
        let w = mitchell((f32(y) - center) / support);
        sum = sum + textureLoad(src_v_plane_rg16, vec2<i32>(i32(gid.x), yc), 0).rg * w;
        weight_sum = weight_sum + w;
    }
    let ws = select(weight_sum, 1.0, abs(weight_sum) < 1e-6);
    let v = clamp(sum / ws, vec2<f32>(0.0), vec2<f32>(1.0));
    textureStore(dst_v_plane_rg16, vec2<i32>(gid.xy), vec4<f32>(v.x, v.y, 0.0, 1.0));
}
