// Cursor overlay pass. Renders a single RGBA sprite at a host-supplied
// position inside the letterboxed video rect, with the same coordinate
// frame the blit pass produces. Runs after the blit (Pass 3) with
// LoadOp::Load and alpha blend so the sprite composites over the
// already-written video pixels without disturbing letterbox bars.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Packed uniform consumed by the cursor pass. Layout matches the Rust
// side's `CursorOverlayUniform` exactly (std140-friendly, all f32):
//
//   uv[0] = (rect_x, rect_y, fit_w, fit_h)    — letterbox rect in window pixels
//   uv[1] = (cursor_x, cursor_y, hotspot_x, hotspot_y) — sprite anchor (video pixels)
//   uv[2] = (cursor_w, cursor_h, video_w, video_h)    — sprite + source dims
//   uv[3] = (surface_w, surface_h, visible, _pad)     — window dims + draw gate
//
// The vertex shader emits a window-NDC quad anchored at
// `rect + (cursor - hotspot) * fit / video` in window-pixel space.
@group(0) @binding(0) var<uniform> u: array<vec4<f32>, 4>;

@group(0) @binding(1) var cursor_tex: texture_2d<f32>;
@group(0) @binding(2) var cursor_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    let rect_xy   = u[0].xy;
    let fit_wh    = u[0].zw;
    let cursor_xy = u[1].xy;
    let hotspot_xy = u[1].zw;
    let cursor_wh = u[2].xy;
    let video_wh  = u[2].zw;
    let surface_wh = u[3].xy;
    let visible = u[3].z;

    // Sprite top-left in window pixels.
    let sprite_origin = rect_xy + (cursor_xy - hotspot_xy) * fit_wh / video_wh;
    // Sprite dims in window pixels — sprite stays in 1:1 capture-pixel
    // scale relative to the video rect, so it doesn't bloat with the
    // Mitchell upscale; that matches what users expect from a cursor.
    let sprite_size = cursor_wh * fit_wh / video_wh;

    // Unit-quad corners with matching uv.
    var corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
    );
    let corner = corners[vi];

    // Window-pixel position → NDC. Hide off-screen when not visible
    // by collapsing the quad to a degenerate point off-screen rather
    // than branching in the fragment shader.
    let px = sprite_origin + corner * sprite_size;
    let ndc_x = (px.x / surface_wh.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (px.y / surface_wh.y) * 2.0;

    var out: VsOut;
    if (visible < 0.5) {
        out.pos = vec4(2.0, 2.0, 0.0, 1.0);
    } else {
        out.pos = vec4(ndc_x, ndc_y, 0.0, 1.0);
    }
    out.uv = corner;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Sprite texture is RGBA8 with straight (non-premultiplied) alpha;
    // the pipeline's blend state does standard SrcAlpha,1-SrcAlpha.
    return textureSample(cursor_tex, cursor_sampler, in.uv);
}
