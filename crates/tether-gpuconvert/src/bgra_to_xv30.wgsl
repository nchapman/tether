// BGRA → XV30 (BT.709 limited-range, 10-bit 4:4:4 packed) conversion.
//
// XV30 (DRM_FORMAT_XV30 / VK_FORMAT_A2B10G10R10_UNORM_PACK32 /
// AV_PIX_FMT_XV30LE) is a single-plane packed 10:10:10:2 layout. Each
// pixel is one little-endian 32-bit word:
//
//   bits [31:30] X (unused / 2-bit padding)
//   bits [29:20] V
//   bits [19:10] U
//   bits [ 9: 0] Y
//
// In Vulkan's A2B10G10R10_UNORM_PACK32 channel mapping (R→[9:0],
// G→[19:10], B→[29:20], A→[31:30]) and the corresponding wgpu
// `Rgb10a2Unorm` storage texture, that lands as:
//
//   R = Y, G = U, B = V, A = X
//
// `textureStore` on `rgb10a2unorm` writes `round(clamp(v, 0, 1) * 1023)`
// for R/G/B and `round(clamp(v, 0, 1) * 3)` for A. The 10-bit value
// lives directly in its 10-bit field — no MSB-align shenanigans like
// P010's 16-bit cells — so the BT.709 limited-range math here uses
// /1023 denominators directly. The math is otherwise identical to
// `bgra_to_p010.wgsl`.
//
// One compute invocation per output pixel; full-resolution UV (no 2x2
// averaging — this is 4:4:4, not 4:2:0).

@group(0) @binding(0) var src: texture_2d<f32>;            // Bgra8Unorm
@group(0) @binding(1) var packed_dst: texture_storage_2d<rgb10a2unorm, write>;

const Y_R: f32 = 0.2126;
const Y_G: f32 = 0.7152;
const Y_B: f32 = 0.0722;
const U_R: f32 = -0.11457;
const U_G: f32 = -0.38543;
const U_B: f32 = 0.50000;
const V_R: f32 = 0.50000;
const V_G: f32 = -0.45415;
const V_B: f32 = -0.04585;

// BT.709 limited-range 10-bit, stored directly in a 10-bit cell:
//   Y_raw  = round(64  + 876 * Y_lin)        ∈ [64, 940]
//   UV_raw = round(512 + 896 * UV_centered)  ∈ [64, 960]
// Storage values are Y_raw / 1023 and UV_raw / 1023.
const Y_SCALE_10:   f32 = 876.0 / 1023.0;
const Y_OFFSET_10:  f32 =  64.0 / 1023.0;
const UV_SCALE_10:  f32 = 896.0 / 1023.0;
const UV_OFFSET_10: f32 = 512.0 / 1023.0;

// Limited-range 10-bit code-value bounds, normalised to the 10-bit
// storage cell. Matches P010's ceiling logic — keep super-white and
// extreme-chroma input in-spec rather than saturating to 0x3FF.
const Y_STORAGE_CEILING:  f32 = 940.0 / 1023.0;
const UV_STORAGE_CEILING: f32 = 960.0 / 1023.0;
const UV_STORAGE_FLOOR:   f32 =  64.0 / 1023.0;

fn rgb_to_y(rgb: vec3<f32>) -> f32 {
    let yp = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    return clamp(yp * Y_SCALE_10 + Y_OFFSET_10, Y_OFFSET_10, Y_STORAGE_CEILING);
}

fn rgb_to_uv(rgb: vec3<f32>) -> vec2<f32> {
    let u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    let v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    let raw = vec2<f32>(u, v) * UV_SCALE_10 + vec2<f32>(UV_OFFSET_10, UV_OFFSET_10);
    return clamp(raw, vec2<f32>(UV_STORAGE_FLOOR), vec2<f32>(UV_STORAGE_CEILING));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(packed_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let rgb = textureLoad(src, coord, 0).rgb;
    let y = rgb_to_y(rgb);
    let uv = rgb_to_uv(rgb);
    // R=Y, G=U, B=V, A=X. The 2-bit A is unused per VAAPI; write 1.0
    // (saturates to 0b11) so unsuspecting readers see a defined value.
    textureStore(packed_dst, coord, vec4<f32>(y, uv.x, uv.y, 1.0));
}
