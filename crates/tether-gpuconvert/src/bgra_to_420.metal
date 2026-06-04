#include <metal_stdlib>
using namespace metal;

constant float Y_R = 0.2126;
constant float Y_G = 0.7152;
constant float Y_B = 0.0722;
constant float U_R = -0.11457;
constant float U_G = -0.38543;
constant float U_B = 0.50000;
constant float V_R = 0.50000;
constant float V_G = -0.45415;
constant float V_B = -0.04585;

constant float Y_SCALE_8 = 219.0 / 255.0;
constant float Y_OFFSET_8 = 16.0 / 255.0;
constant float UV_SCALE_8 = 224.0 / 255.0;
constant float UV_OFFSET_8 = 128.0 / 255.0;

constant float Y_SCALE_10 = 56064.0 / 65535.0;
constant float Y_OFFSET_10 = 4096.0 / 65535.0;
constant float UV_SCALE_10 = 57344.0 / 65535.0;
constant float UV_OFFSET_10 = 32768.0 / 65535.0;
constant float Y_STORAGE_CEILING = 60160.0 / 65535.0;
constant float UV_STORAGE_CEILING = 61440.0 / 65535.0;
constant float UV_STORAGE_FLOOR = 4096.0 / 65535.0;

static inline float rgb_to_y8(float3 rgb) {
    float yp = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    return yp * Y_SCALE_8 + Y_OFFSET_8;
}

static inline float2 rgb_to_uv8(float3 rgb) {
    float u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    float v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    return float2(u, v) * UV_SCALE_8 + float2(UV_OFFSET_8, UV_OFFSET_8);
}

static inline float rgb_to_y10(float3 rgb) {
    float yp = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    return clamp(yp * Y_SCALE_10 + Y_OFFSET_10, Y_OFFSET_10, Y_STORAGE_CEILING);
}

static inline float2 rgb_to_uv10(float3 rgb) {
    float u = U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    float v = V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    float2 raw = float2(u, v) * UV_SCALE_10 + float2(UV_OFFSET_10, UV_OFFSET_10);
    return clamp(raw, float2(UV_STORAGE_FLOOR), float2(UV_STORAGE_CEILING));
}

static inline uint2 clamp_coord(uint2 coord, uint width, uint height) {
    return uint2(min(coord.x, width - 1), min(coord.y, height - 1));
}

kernel void bgra_to_420v(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> y_dst [[texture(1)]],
                         texture2d<float, access::write> uv_dst [[texture(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uv_dst.get_width() || gid.y >= uv_dst.get_height()) {
        return;
    }

    uint lx = gid.x * 2;
    uint ly = gid.y * 2;
    uint width = y_dst.get_width();
    uint height = y_dst.get_height();

    float3 p00 = src.read(uint2(lx, ly)).rgb;
    float3 p10 = src.read(clamp_coord(uint2(lx + 1, ly), width, height)).rgb;
    float3 p01 = src.read(clamp_coord(uint2(lx, ly + 1), width, height)).rgb;
    float3 p11 = src.read(clamp_coord(uint2(lx + 1, ly + 1), width, height)).rgb;

    y_dst.write(float4(rgb_to_y8(p00), 0.0, 0.0, 1.0), uint2(lx, ly));
    if (lx + 1 < width) {
        y_dst.write(float4(rgb_to_y8(p10), 0.0, 0.0, 1.0), uint2(lx + 1, ly));
    }
    if (ly + 1 < height) {
        y_dst.write(float4(rgb_to_y8(p01), 0.0, 0.0, 1.0), uint2(lx, ly + 1));
        if (lx + 1 < width) {
            y_dst.write(float4(rgb_to_y8(p11), 0.0, 0.0, 1.0), uint2(lx + 1, ly + 1));
        }
    }

    float3 chroma_rgb = (p00 + p10 + p01 + p11) * 0.25;
    float2 uv = rgb_to_uv8(chroma_rgb);
    uv_dst.write(float4(uv.x, uv.y, 0.0, 1.0), gid);
}

kernel void bgra_to_x420(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> y_dst [[texture(1)]],
                         texture2d<float, access::write> uv_dst [[texture(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uv_dst.get_width() || gid.y >= uv_dst.get_height()) {
        return;
    }

    uint lx = gid.x * 2;
    uint ly = gid.y * 2;
    uint width = y_dst.get_width();
    uint height = y_dst.get_height();

    float3 p00 = src.read(uint2(lx, ly)).rgb;
    float3 p10 = src.read(clamp_coord(uint2(lx + 1, ly), width, height)).rgb;
    float3 p01 = src.read(clamp_coord(uint2(lx, ly + 1), width, height)).rgb;
    float3 p11 = src.read(clamp_coord(uint2(lx + 1, ly + 1), width, height)).rgb;

    y_dst.write(float4(rgb_to_y10(p00), 0.0, 0.0, 1.0), uint2(lx, ly));
    if (lx + 1 < width) {
        y_dst.write(float4(rgb_to_y10(p10), 0.0, 0.0, 1.0), uint2(lx + 1, ly));
    }
    if (ly + 1 < height) {
        y_dst.write(float4(rgb_to_y10(p01), 0.0, 0.0, 1.0), uint2(lx, ly + 1));
        if (lx + 1 < width) {
            y_dst.write(float4(rgb_to_y10(p11), 0.0, 0.0, 1.0), uint2(lx + 1, ly + 1));
        }
    }

    float3 chroma_rgb = (p00 + p10 + p01 + p11) * 0.25;
    float2 uv = rgb_to_uv10(chroma_rgb);
    uv_dst.write(float4(uv.x, uv.y, 0.0, 1.0), gid);
}
