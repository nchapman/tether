// BGRA → packed YUV 4:4:4 (DRM_FORMAT_XYUV8888 / AV_PIX_FMT_VUYX,
// BT.709 limited-range).
//
// Why packed (not planar): ffmpeg 8.x's `vaapi_drm_format_map` has no
// entry for planar YUV444P over DRM_PRIME — `av_hwframe_map(DRM_PRIME
// → VAAPI)` fails with "DRM format not supported by VAAPI" when fed a
// three-R8-plane descriptor with the YU24 aggregate fourcc. The packed
// XYUV8888 layout IS in the table and maps to VA_FOURCC_XYUV /
// AV_PIX_FMT_VUYX, which Intel/AMD VAAPI encoders accept as input to
// HEVC Main 4:4:4. One 32-bit-per-pixel plane in memory:
//
//   DRM_FORMAT_XYUV8888 little-endian byte order: [V, U, Y, X]
//   ↔ wgpu Rgba8Unorm channels with .r=V, .g=U, .b=Y, .a=X
//
// At ~2× the per-pixel bytes of NV12 the bandwidth cost is real
// (~5.5 MB/frame at 1080p), but the encoder reads it once per frame
// off LPDDR — nowhere near a bottleneck at 60 fps.
//
// BT.709 limited-range matrix kept in sync with the NV12 sibling
// shader. Any divergence here would show up as a colour shift on
// 4:4:4 sessions only — easy to miss in code review.

@group(0) @binding(0) var src: texture_2d<f32>;             // Bgra8Unorm
@group(0) @binding(1) var packed_dst: texture_storage_2d<rgba8unorm, write>;

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

// 8x8 workgroup, one invocation per output pixel. No chroma collapse.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(packed_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = textureLoad(src, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb;
    let y = rgb_to_y(p);
    let u = rgb_to_u(p);
    let v = rgb_to_v(p);
    // DRM_FORMAT_XYUV8888 byte 0 = V, byte 1 = U, byte 2 = Y, byte 3 = X.
    // Rgba8Unorm channel-to-byte mapping is .r=byte0, .g=byte1, .b=byte2,
    // .a=byte3. So we write (V, U, Y, 1.0) → memory becomes [V, U, Y, 0xFF].
    textureStore(
        packed_dst,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(v, u, y, 1.0),
    );
}
