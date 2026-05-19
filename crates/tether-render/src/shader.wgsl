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

// BT.709 limited-range YUV -> linear sRGB. Constants match the
// conventional broadcast matrix. The encoder is currently configured
// for Bt709Limited so this is right; full-range or BT.601 sources
// would need a different matrix (deferred until we negotiate
// per-stream color metadata).
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let y = textureSample(y_tex, s, in.uv).r;
    let chroma = textureSample(uv_tex, s, in.uv).rg;
    // Limited-range expansion: Y[16..235] -> [0..1], C[16..240] -> [-0.5..0.5].
    let yc = (y - 16.0 / 255.0) * (255.0 / 219.0);
    let uc = (chroma.r - 128.0 / 255.0) * (255.0 / 224.0);
    let vc = (chroma.g - 128.0 / 255.0) * (255.0 / 224.0);
    let r = yc + 1.5748 * vc;
    let g = yc - 0.1873 * uc - 0.4681 * vc;
    let b = yc + 1.8556 * uc;
    return vec4(r, g, b, 1.0);
}
