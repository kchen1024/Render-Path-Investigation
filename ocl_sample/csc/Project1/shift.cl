__kernel void kernel_bayer_shift (
    __read_only image2d_t input, __write_only image2d_t output)
{

    int x = get_global_id (0);
    int y = get_global_id (1);
    float4 pixel = read_imagef(input, (int2)(x, y));

    float4 pixel_out;
	
    // Extract 12-bit and 4-bit components
    ushort value_4bitlow = pixel.x & 0xF;
    ushort value_12bithigh = pixel.x >> 4;
	pixel_out.x = (value_12bithigh << 4) | value_4bitlow;
	
	value_4bitlow = pixel.y & 0xF;
	value_12bithigh = pixel.y >> 4;
	pixel_out.y = (value_12bithigh << 4) | value_4bitlow;

	value_4bitlow = pixel.z & 0xF;
	value_12bithigh = pixel.z >> 4;
	pixel_out.z = (value_12bithigh << 4) | value_4bitlow;
	
	value_4bitlow = pixel.w & 0xF;
	value_12bithigh = pixel.w >> 4;
	pixel_out.w = (value_12bithigh << 4) | value_4bitlow;

    write_imagef(output, (int2)(x, y), pixel_out);

}