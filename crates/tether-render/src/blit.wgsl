// Final swapchain blit. Samples an Rgba16Float linear-light texture
// (either the YUV->RGB intermediate at video dims, or the Mitchell
// scaler's output at letterbox-fit dims) and writes to the swapchain
// with aspect-preserving letterbox.
//
// The output is linear light; the swapchain format applies the OETF
// on write (sRGB encoding for an Rgba8UnormSrgb / Bgra8UnormSrgb
// surface), same convention as the existing YUV->RGB pass.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// `.xy` = (x_scale, y_scale) for aspect-ratio preserving letterbox /
// pillarbox, identical convention to shader.wgsl. `.zw` padded out.
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

@group(0) @binding(0) var rgb_tex: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(rgb_tex, s, in.uv);
}
