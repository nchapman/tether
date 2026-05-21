// BGRA → P410 (BT.709 limited-range, 10-bit 4:4:4) conversion.
//
// P410 layout is NV24 with the cell width doubled: full-resolution
// 16-bit Y plane plus full-resolution 16-bit interleaved UV plane.
// 10-bit data is MSB-aligned (bits [15:6]) in each cell — same
// convention as P010 (see bgra_to_p010.wgsl for the range derivation).
//
// One compute invocation per output pixel; each writes one Y sample
// and one UV pair. No chroma subsampling — full-res UV means no 2x2
// averaging.

@group(0) @binding(0) var src: texture_2d<f32>;            // Bgra8Unorm
@group(0) @binding(1) var y_dst: texture_storage_2d<r16unorm, write>;
@group(0) @binding(2) var uv_dst: texture_storage_2d<rg16unorm, write>;

const Y_R: f32 = 0.2126;
const Y_G: f32 = 0.7152;
const Y_B: f32 = 0.0722;
const U_R: f32 = -0.11457;
const U_G: f32 = -0.38543;
const U_B: f32 = 0.50000;
const V_R: f32 = 0.50000;
const V_G: f32 = -0.45415;
const V_B: f32 = -0.04585;

// 10-bit MSB-aligned scale/offset (see bgra_to_p010.wgsl for the
// derivation — identical constants because the storage convention is
// the same; only the per-plane resolution differs).
const Y_SCALE_10:  f32 = 56064.0 / 65535.0;
const Y_OFFSET_10: f32 =  4096.0 / 65535.0;
const UV_SCALE_10:  f32 = 57344.0 / 65535.0;
const UV_OFFSET_10: f32 = 32768.0 / 65535.0;

// MSB-aligned 10-bit cell ceilings — see bgra_to_p010.wgsl for the
// derivation. Clamping ensures storage values have bits [5:0] zero
// instead of saturating to 0xFFFF (low 6 bits set) on super-white
// or extreme-chroma input — out-of-spec for MSB-aligned 10-bit.
const Y_STORAGE_CEILING:  f32 = 60160.0 / 65535.0;
const UV_STORAGE_CEILING: f32 = 61440.0 / 65535.0;
const UV_STORAGE_FLOOR:   f32 =  4096.0 / 65535.0;

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
    let dims = textureDimensions(y_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let rgb = textureLoad(src, coord, 0).rgb;
    textureStore(y_dst, coord, vec4<f32>(rgb_to_y(rgb), 0.0, 0.0, 1.0));
    let uv = rgb_to_uv(rgb);
    textureStore(uv_dst, coord, vec4<f32>(uv, 0.0, 1.0));
}
