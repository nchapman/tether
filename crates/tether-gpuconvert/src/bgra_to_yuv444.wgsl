// BGRA → YUV 4:4:4 planar (BT.709 limited-range) conversion.
//
// Each compute invocation processes one BGRA pixel and writes one Y, one
// U, and one V sample. No chroma subsampling — U and V are full
// resolution, matching what HEVC Main444 / `AV_PIX_FMT_YUV444P` expects
// over the wire. This is the path that preserves text antialiasing and
// saturated UI colour accents that 4:2:0 visibly smears.
//
// Why three R8 storage targets (not one Rgba8 packed):
//   - libavcodec's `AV_PIX_FMT_YUV444P` is strictly planar with separate
//     stride per plane. Packing into a single image would require an
//     additional CPU-side deinterleave the dma-buf path is designed to
//     avoid.
//   - VAAPI's DRM_PRIME importer accepts DRM_FORMAT_YUV444 as a
//     three-plane layout (one DRM object with three offsets); this
//     shader writes directly into that shape.
//
// BT.709 limited-range — same matrix as `bgra_to_nv12.wgsl`. Keep the
// constants in sync between the two shaders; the encoder VUI is shared,
// so a divergence here would produce a colour shift visible only on
// 4:4:4 sessions.

@group(0) @binding(0) var src: texture_2d<f32>;          // Bgra8Unorm
@group(0) @binding(1) var y_dst: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var u_dst: texture_storage_2d<r8unorm, write>;
@group(0) @binding(3) var v_dst: texture_storage_2d<r8unorm, write>;

const Y_R: f32 = 0.2126;
const Y_G: f32 = 0.7152;
const Y_B: f32 = 0.0722;
const U_R: f32 = -0.11457;
const U_G: f32 = -0.38543;
const U_B: f32 = 0.50000;
const V_R: f32 = 0.50000;
const V_G: f32 = -0.45415;
const V_B: f32 = -0.04585;

const Y_SCALE: f32 = 219.0 / 255.0;
const Y_OFFSET: f32 = 16.0 / 255.0;
const UV_SCALE: f32 = 224.0 / 255.0;
const UV_OFFSET: f32 = 128.0 / 255.0;

fn rgb_to_y(rgb: vec3<f32>) -> f32 {
    let yp = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    return yp * Y_SCALE + Y_OFFSET;
}

fn rgb_to_u(rgb: vec3<f32>) -> f32 {
    let u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    return u * UV_SCALE + UV_OFFSET;
}

fn rgb_to_v(rgb: vec3<f32>) -> f32 {
    let v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    return v * UV_SCALE + UV_OFFSET;
}

// 8x8 workgroup, one invocation per output pixel. No 2x2 collapse —
// chroma is full-res in 4:4:4, so each invocation is one Y / one U /
// one V sample.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(y_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = textureLoad(src, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb;
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(y_dst, coord, vec4<f32>(rgb_to_y(p), 0.0, 0.0, 1.0));
    textureStore(u_dst, coord, vec4<f32>(rgb_to_u(p), 0.0, 0.0, 1.0));
    textureStore(v_dst, coord, vec4<f32>(rgb_to_v(p), 0.0, 0.0, 1.0));
}
