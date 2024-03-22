__kernel void nv12_to_argb(__read_only image2d_t src_y,
                           __write_only image2d_t dst_argb,
                           __read_only image2d_t src_uv,
                           const int width,
                           const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Read Y value from src_y
    float y_value = read_imagef(src_y, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE, (int2)(x, y)).x;

    // Read UV value from src_uv
    float2 uv_value = read_imagef(src_uv, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE, (int2)(x / 2, y / 2)).xy;

    // NV12 to ARGB conversion
    float y = y_value * 255.0f;
    float u = uv_value.x * 255.0f - 128.0f;
    float v = uv_value.y * 255.0f - 128.0f;

    float r = y + 1.402f * v;
    float g = y - 0.344136f * u - 0.714136f * v;
    float b = y + 1.772f * u;

    // Clamp the values to [0, 255]
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);

    // Write ARGB value to dst_argb
    write_imagef(dst_argb, (int2)(x, y), (float4)(r/255.0f, g/255.0f, b/255.0f, 1.0f));
}