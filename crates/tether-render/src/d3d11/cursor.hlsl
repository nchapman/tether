// D3D11 cursor overlay pass. Line-for-line port of `cursor.wgsl`:
// renders a single straight-alpha RGBA sprite at a host-supplied
// position inside the letterboxed video rect, with the same coordinate
// frame the YUV blit produces. Runs after the video draw with the RTV
// loaded (not cleared) and an alpha-blend state set, so the sprite
// composites over the already-written video pixels without disturbing
// the letterbox bars.
//
// The packed `Params` cbuffer matches the bytes `cursor_uniform_for`
// emits (4 × float4, std140-equivalent; HLSL packs each float4[] element
// in its own 16-byte register):
//
//   u[0] = (rect_x, rect_y, fit_w, fit_h)              — letterbox rect in window pixels
//   u[1] = (cursor_x, cursor_y, hotspot_x, hotspot_y)  — sprite anchor (video pixels)
//   u[2] = (cursor_w, cursor_h, video_w, video_h)      — sprite + source dims
//   u[3] = (surface_w, surface_h, visible, _pad)       — window dims + draw gate

cbuffer Params : register(b0) {
    float4 u[4];
};

Texture2D cursor_tex : register(t0);
SamplerState cursor_sampler : register(s0);

struct VsOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VsOut vs(uint vid : SV_VertexID) {
    float2 rect_xy = u[0].xy;
    float2 fit_wh = u[0].zw;
    float2 cursor_xy = u[1].xy;
    float2 hotspot_xy = u[1].zw;
    float2 cursor_wh = u[2].xy;
    float2 video_wh = u[2].zw;
    float2 surface_wh = u[3].xy;
    float visible = u[3].z;

    // Sprite top-left in window pixels.
    float2 sprite_origin = rect_xy + (cursor_xy - hotspot_xy) * fit_wh / video_wh;
    // Sprite dims in window pixels — sprite stays in 1:1 capture-pixel
    // scale relative to the video rect, so it doesn't bloat with the
    // Mitchell upscale; that matches what users expect from a cursor.
    float2 sprite_size = cursor_wh * fit_wh / video_wh;

    // Unit-quad corners with matching uv (two triangles).
    float2 corners[6] = {
        float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
        float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0),
    };
    float2 corner = corners[vid];

    // Window-pixel position → NDC. Hide when not visible by collapsing
    // the quad to an off-screen point rather than branching in the
    // pixel shader.
    float2 px = sprite_origin + corner * sprite_size;
    float ndc_x = (px.x / surface_wh.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (px.y / surface_wh.y) * 2.0;

    VsOut o;
    if (visible < 0.5) {
        o.pos = float4(2.0, 2.0, 0.0, 1.0);
    } else {
        o.pos = float4(ndc_x, ndc_y, 0.0, 1.0);
    }
    o.uv = corner;
    return o;
}

float4 ps(VsOut input) : SV_Target {
    // Sprite texture is RGBA8 straight (non-premultiplied) alpha; the
    // blend state does standard SrcAlpha, 1-SrcAlpha. Sampled non-sRGB
    // (matching wgpu's Rgba8Unorm cursor texture) so the composite over
    // the sRGB render target matches the Linux/macOS path exactly.
    return cursor_tex.Sample(cursor_sampler, input.uv);
}
