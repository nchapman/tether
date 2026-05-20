// BGRA → NV12 (BT.709 limited-range) conversion.
//
// Each compute invocation processes a 2x2 block of input BGRA pixels and
// writes:
//   - 4 Y samples (one per input pixel) to the Y plane (full resolution)
//   - 1 averaged UV pair to the UV plane (half resolution in each axis,
//     matching 4:2:0 chroma subsampling)
//
// Why BT.709 limited-range:
//   - >720p is the dominant screen-capture resolution; BT.709 is the
//     standard for HD content.
//   - Limited range (Y∈[16,235], UV∈[16,240]) is what most H.264
//     decoders default to when the VUI doesn't say otherwise, matching
//     consumer-video pipelines. The encoder sets VUI flags accordingly.
//   - BT.601 / BT.2020 land as separate shader variants if/when needed.

@group(0) @binding(0) var src: texture_2d<f32>;          // Bgra8Unorm
@group(0) @binding(1) var y_dst: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_dst: texture_storage_2d<rg8unorm, write>;

// BT.709 limited-range coefficients. RGB inputs are in [0, 1] (the
// hardware unorm sampling does the divide-by-255). Y output is in
// [16/255, 235/255]; UV outputs are in [16/255, 240/255].
//
// Derivation (so the next reviewer doesn't re-derive and miscalculate):
//   Kr = 0.2126, Kg = 0.7152, Kb = 0.0722  (BT.709 luma weights)
//   Y' = Kr*R + Kg*G + Kb*B
//   Cb = (B - Y') / (2 * (1 - Kb))   ->  denom = 1.8556
//        =  -Kr/d   * R  +  -Kg/d   * G  +  (1-Kb)/d * B
//        =  -0.11457 R   +  -0.38543 G   +   0.50000 B
//   Cr = (R - Y') / (2 * (1 - Kr))   ->  denom = 1.5748
//        =  (1-Kr)/d * R  +  -Kg/d   * G  +  -Kb/d   * B
//        =   0.50000 R    +  -0.45415 G   +  -0.04585 B
// Source: ITU-R BT.709-6 Annex B (also Wikipedia "YCbCr#ITU-R_BT.709").
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

fn rgb_to_uv(rgb: vec3<f32>) -> vec2<f32> {
    let u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    let v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    return vec2<f32>(u, v) * UV_SCALE + vec2<f32>(UV_OFFSET, UV_OFFSET);
}

// Workgroup of 8x8 invocations, each handling a 2x2 input block →
// covers 16x16 input pixels per workgroup. Tuned for cache friendliness
// rather than max occupancy; the dispatcher rounds up to fit.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid maps to chroma-pixel coordinates (each invocation is one UV
    // sample's worth of work, covering a 2x2 luma block).
    let chroma_dims = textureDimensions(uv_dst);
    if (gid.x >= chroma_dims.x || gid.y >= chroma_dims.y) {
        return;
    }

    let lx = i32(gid.x * 2u);
    let ly = i32(gid.y * 2u);

    let luma_dims = textureDimensions(y_dst);

    // Sample the 2x2 input block. We clamp the addresses against the
    // luma plane's dimensions so odd resolutions (e.g. 1919×1079 — rare
    // but possible from PipeWire's "Region" capture) replicate the edge
    // pixel instead of sampling out-of-bounds.
    let p00 = textureLoad(src, vec2<i32>(lx,                          ly                          ), 0).rgb;
    let p10 = textureLoad(src, vec2<i32>(min(lx + 1, i32(luma_dims.x) - 1), ly                          ), 0).rgb;
    let p01 = textureLoad(src, vec2<i32>(lx,                          min(ly + 1, i32(luma_dims.y) - 1)), 0).rgb;
    let p11 = textureLoad(src, vec2<i32>(min(lx + 1, i32(luma_dims.x) - 1), min(ly + 1, i32(luma_dims.y) - 1)), 0).rgb;

    // Write the four Y samples. Bounds-check each in case the luma
    // plane width/height is odd.
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

    // 2x2 box-average for chroma, then convert. Box-averaging in RGB
    // before conversion matches what swscale does by default for 4:2:0
    // subsampling — cheap, predictable, and the same shape pictures
    // most consumer decoders post-process for.
    let chroma_rgb = (p00 + p10 + p01 + p11) * 0.25;
    let uv = rgb_to_uv(chroma_rgb);
    textureStore(uv_dst, vec2<i32>(gid.xy), vec4<f32>(uv, 0.0, 1.0));
}
