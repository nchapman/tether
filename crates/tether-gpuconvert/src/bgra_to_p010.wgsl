// BGRA → P010 (BT.709 limited-range, 10-bit 4:2:0) conversion.
//
// P010 layout is NV12 with the cell width doubled: full-resolution
// 16-bit Y plane plus half-resolution 16-bit interleaved UV plane.
// The 10-bit sample lives in bits [15:6] of each 16-bit cell with the
// low 6 bits zero. This is the convention FFmpeg's vaapi pix_fmt P010LE
// expects on the encoder side, and the convention the renderer's
// `luma_scale = 65535 / 65472` factor compensates for on the decoder
// side.
//
// Each compute invocation processes a 2x2 BGRA block and writes:
//   - 4 Y samples (one per input pixel) to the Y plane (R16Unorm)
//   - 1 averaged UV pair to the UV plane (Rg16Unorm, half-res)
//
// Range math is derived from the 10-bit ITU-R BT.709 spec rather than
// the 8-bit one: limited-range Y is [64, 940] in raw 10-bit, [4096,
// 60160] in MSB-aligned 16-bit storage. Cb/Cr is [64, 960] raw, [4096,
// 61440] MSB-aligned. We compute the normalised storage value directly
// so a single texture-store writes the right MSB-aligned bytes
// (textureStore on R16Unorm writes round(value * 65535) to the cell).

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

// 10-bit MSB-aligned scale/offset for textureStore on R16/Rg16Unorm:
//   Y storage = (Y' * 876 + 64) * 64 / 65535
//             = Y' * 56064/65535 + 4096/65535
//   UV storage = (UV_centered * 896 + 512) * 64 / 65535
//              = UV_centered * 57344/65535 + 32768/65535
const Y_SCALE_10:  f32 = 56064.0 / 65535.0;
const Y_OFFSET_10: f32 =  4096.0 / 65535.0;
const UV_SCALE_10:  f32 = 57344.0 / 65535.0;
const UV_OFFSET_10: f32 = 32768.0 / 65535.0;

fn rgb_to_y(rgb: vec3<f32>) -> f32 {
    let yp = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    return yp * Y_SCALE_10 + Y_OFFSET_10;
}

fn rgb_to_uv(rgb: vec3<f32>) -> vec2<f32> {
    let u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    let v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    return vec2<f32>(u, v) * UV_SCALE_10 + vec2<f32>(UV_OFFSET_10, UV_OFFSET_10);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chroma_dims = textureDimensions(uv_dst);
    if (gid.x >= chroma_dims.x || gid.y >= chroma_dims.y) {
        return;
    }

    let lx = i32(gid.x * 2u);
    let ly = i32(gid.y * 2u);
    let luma_dims = textureDimensions(y_dst);

    // Clamped 2x2 sample — odd dims replicate the edge pixel, matching
    // the 8-bit NV12 path's behaviour.
    let p00 = textureLoad(src, vec2<i32>(lx,                              ly                              ), 0).rgb;
    let p10 = textureLoad(src, vec2<i32>(min(lx + 1, i32(luma_dims.x) - 1), ly                              ), 0).rgb;
    let p01 = textureLoad(src, vec2<i32>(lx,                              min(ly + 1, i32(luma_dims.y) - 1)), 0).rgb;
    let p11 = textureLoad(src, vec2<i32>(min(lx + 1, i32(luma_dims.x) - 1), min(ly + 1, i32(luma_dims.y) - 1)), 0).rgb;

    textureStore(y_dst, vec2<i32>(lx,     ly    ), vec4<f32>(rgb_to_y(p00), 0.0, 0.0, 1.0));
    if (lx + 1 < i32(luma_dims.x)) {
        textureStore(y_dst, vec2<i32>(lx + 1, ly    ), vec4<f32>(rgb_to_y(p10), 0.0, 0.0, 1.0));
    }
    if (ly + 1 < i32(luma_dims.y)) {
        textureStore(y_dst, vec2<i32>(lx,     ly + 1), vec4<f32>(rgb_to_y(p01), 0.0, 0.0, 1.0));
        if (lx + 1 < i32(luma_dims.x)) {
            textureStore(y_dst, vec2<i32>(lx + 1, ly + 1), vec4<f32>(rgb_to_y(p11), 0.0, 0.0, 1.0));
        }
    }

    // Box-averaged chroma per 2x2 block. Same shape as the 8-bit
    // shader; predictable, cheap, and matches what swscale does by
    // default for 4:2:0 subsampling.
    let chroma_rgb = (p00 + p10 + p01 + p11) * 0.25;
    let uv = rgb_to_uv(chroma_rgb);
    textureStore(uv_dst, vec2<i32>(gid.xy), vec4<f32>(uv, 0.0, 1.0));
}
