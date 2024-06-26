/*
 * function: kernel_csc_nv12torgba
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * vertical_offset, vertical offset from y to uv
 */

__kernel void kernel_csc_nv12torgba (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_y1 = read_imagef(input, sampler, (int2)(2 * x, 2 * y));
    float4 pixel_y2 = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y));
    float4 pixel_y3 = read_imagef(input, sampler, (int2)(2 * x, 2 * y + 1));
    float4 pixel_y4 = read_imagef(input, sampler, (int2)(2 * x + 1, 2 * y + 1));
    float4 pixel_u = read_imagef(input, sampler, (int2)(2 * x, y + vertical_offset));
    float4 pixel_v = read_imagef(input, sampler, (int2)(2 * x + 1, y + vertical_offset));
    float4 pixel_out1, pixel_out2, pixel_out3, pixel_out4;
    pixel_out1.x = pixel_y1.x + 1.13983 * (pixel_v.x - 0.5);
    pixel_out1.y = pixel_y1.x - 0.39465 * (pixel_u.x - 0.5) - 0.5806 * (pixel_v.x - 0.5);
    pixel_out1.z = pixel_y1.x + 2.03211 * (pixel_u.x - 0.5);
    pixel_out1.w = 0.0;
    pixel_out2.x = pixel_y2.x + 1.13983 * (pixel_v.x - 0.5);
    pixel_out2.y = pixel_y2.x - 0.39465 * (pixel_u.x - 0.5) - 0.5806 * (pixel_v.x - 0.5);
    pixel_out2.z = pixel_y2.x + 2.03211 * (pixel_u.x - 0.5);
    pixel_out2.w = 0.0;
    pixel_out3.x = pixel_y3.x + 1.13983 * (pixel_v.x - 0.5);
    pixel_out3.y = pixel_y3.x - 0.39465 * (pixel_u.x - 0.5) - 0.5806 * (pixel_v.x - 0.5);
    pixel_out3.z = pixel_y3.x + 2.03211 * (pixel_u.x - 0.5);
    pixel_out3.w = 0.0;
    pixel_out4.x = pixel_y4.x + 1.13983 * (pixel_v.x - 0.5);
    pixel_out4.y = pixel_y4.x - 0.39465 * (pixel_u.x - 0.5) - 0.5806 * (pixel_v.x - 0.5);
    pixel_out4.z = pixel_y4.x + 2.03211 * (pixel_u.x - 0.5);
    pixel_out4.w = 0.0;
    write_imagef(output, (int2)(2 * x, 2 * y), pixel_out1);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), pixel_out2);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), pixel_out3);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), pixel_out4);
}