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

// YUV 4:2:0 planar: three single-channel textures, full-res Y plus
// half-res U + V. The sampler bilerps the chroma planes for us — at
// 4:2:0 quality the visual difference vs. nearest is negligible and
// the bilinear taps are free on every GPU we care about.
@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var s: sampler;

// BT.709 limited-range YUV -> linear sRGB. Constants match the
// conventional broadcast matrix. The encoder is currently configured
// for Bt709Limited so this is right; full-range or BT.601 sources
// would need a different matrix (deferred until we negotiate
// per-stream color metadata).
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let y = textureSample(y_tex, s, in.uv).r;
    let u = textureSample(u_tex, s, in.uv).r;
    let v = textureSample(v_tex, s, in.uv).r;
    // Limited-range expansion: Y[16..235] -> [0..1], C[16..240] -> [-0.5..0.5].
    let yc = (y - 16.0 / 255.0) * (255.0 / 219.0);
    let uc = (u - 128.0 / 255.0) * (255.0 / 224.0);
    let vc = (v - 128.0 / 255.0) * (255.0 / 224.0);
    let r = yc + 1.5748 * vc;
    let g = yc - 0.1873 * uc - 0.4681 * vc;
    let b = yc + 1.8556 * uc;
    return vec4(r, g, b, 1.0);
}
