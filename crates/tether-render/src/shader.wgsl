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

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
