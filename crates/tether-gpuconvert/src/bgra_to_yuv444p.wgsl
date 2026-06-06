// BGRA -> planar YUV444P (DRM_FORMAT_YUV444 / YU24, BT.709 limited-range).
//
// NVIDIA's NVENC 4:4:4 path consumes AV_PIX_FMT_YUV444P: three full-size
// 8-bit planes. This shader writes Y, U, and V into separate R8 storage
// textures with no chroma subsampling.

@group(0) @binding(0) var src: texture_2d<f32>; // Bgra8Unorm
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(y_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let rgb = textureLoad(src, xy, 0).rgb;
    textureStore(y_dst, xy, vec4<f32>(rgb_to_y(rgb), 0.0, 0.0, 1.0));
    textureStore(u_dst, xy, vec4<f32>(rgb_to_u(rgb), 0.0, 0.0, 1.0));
    textureStore(v_dst, xy, vec4<f32>(rgb_to_v(rgb), 0.0, 0.0, 1.0));
}
