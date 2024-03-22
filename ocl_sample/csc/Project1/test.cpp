#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.hpp>
#include <string.h>
#include <iostream>
#include <vector>

#define CHECK_CL(err) \
    if (err != CL_SUCCESS) { \
        printf("OpenCL error %d\n", err); \
        return EXIT_FAILURE; \
    }

static const char kernelString[] = R"CLC(
__kernel void kernel_csc_nv12torgba (
    __read_only image2d_t input_y, __write_only image2d_t output, __read_only image2d_t input_uv)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    float4 pixel_y1 = read_imagef(input_y, (int2)(2 * x, 2 * y));
    float4 pixel_y2 = read_imagef(input_y, (int2)(2 * x + 1, 2 * y));
    float4 pixel_y3 = read_imagef(input_y, (int2)(2 * x, 2 * y + 1));
    float4 pixel_y4 = read_imagef(input_y, (int2)(2 * x + 1, 2 * y + 1));
    float4 pixel_u = read_imagef(input_uv, (int2)(2 * x, y));
    float4 pixel_v = read_imagef(input_uv, (int2)(2 * x + 1, y));
    float4 pixel_out1, pixel_out2, pixel_out3, pixel_out4;
    pixel_out1.x = pixel_y1.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out1.y = pixel_y1.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out1.z = pixel_y1.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out1.w = 0.0f;
    pixel_out2.x = pixel_y2.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out2.y = pixel_y2.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out2.z = pixel_y2.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out2.w = 0.0f;
    pixel_out3.x = pixel_y3.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out3.y = pixel_y3.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out3.z = pixel_y3.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out3.w = 0.0f;
    pixel_out4.x = pixel_y4.x + 1.13983f * (pixel_v.x - 0.5f);
    pixel_out4.y = pixel_y4.x - 0.39465f * (pixel_u.x - 0.5f) - 0.5806f * (pixel_v.x - 0.5f);
    pixel_out4.z = pixel_y4.x + 2.03211f * (pixel_u.x - 0.5f);
    pixel_out4.w = 0.0f;
    write_imagef(output, (int2)(2 * x, 2 * y), pixel_out1);
    write_imagef(output, (int2)(2 * x + 1, 2 * y), pixel_out2);
    write_imagef(output, (int2)(2 * x, 2 * y + 1), pixel_out3);
    write_imagef(output, (int2)(2 * x + 1, 2 * y + 1), pixel_out4);
}
)CLC";

int LoadData(void* buffer, size_t size, const char* filename, uint32_t offset)
{
    FILE* hFile = NULL;
    hFile = fopen(filename, "rb");
    if (hFile == NULL)
    {
        printf(">> Open the file %s failed. \n", filename);
        return -1;
    }

    if (offset > 0)
    {
        _fseeki64(hFile, offset, 0);
    }

    if (fread(buffer, 1, size, hFile) != size)
    {
        printf(">> Read the file %s data failed. \n", filename);
        return -2;
    }
    fclose(hFile);
    return 0;
}

void SplitNV12(const char* filename, int width, int height, unsigned char* yPlane, unsigned char* uvPlane) {

    size_t inSize = width * height * 3 / 2;
    unsigned char* nv12Data = (unsigned char*)malloc(inSize);
    LoadData(nv12Data, inSize, filename, 0);
    int i, j, uvIndex;

    // Copy Y plane
    memcpy(yPlane, nv12Data, width * height);

    // Copy UV plane
    uvIndex = width * height;
    for (i = 0, j = 0; i < width * height / 4; i++, j += 2) {
        uvPlane[i * 2] = nv12Data[uvIndex + j];       // U
        uvPlane[i * 2 + 1] = nv12Data[uvIndex + j + 1]; // V
    }
}

int DumpData(void* buffer, size_t size, const char* fileName)
{
    FILE* hFile = NULL;
    fopen_s(&hFile, fileName, "wb");
    if (hFile == NULL)
    {
        printf("Open the file: %s failed.\n", fileName);
        return -1;
    }
    int buff = fwrite(buffer, 1, size, hFile);
    if (buff != size)
    {
        printf("Write the file %s data failed\n", fileName);
        return -2;
    }
    fclose(hFile);

    return 0;
}


int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;
    bool bDump = true;
    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());

    cl::Context context{ devices[deviceIndex] };
    cl::CommandQueue commandQueue{ context, devices[deviceIndex] };

    cl::Program program{ context, kernelString };
    err = program.build();
    CHECK_CL(err);

    cl::Kernel kernelCsc = cl::Kernel{ program, "kernel_csc_nv12torgba" };

    int width = 3840;
    int height = 2160;
    unsigned char* yPlane = (unsigned char*)malloc(width * height);
    unsigned char* uvPlane = (unsigned char*)malloc(width * height / 2);
    SplitNV12("3840x2160.nv12", width, height, yPlane, uvPlane);

    cl::ImageFormat imgformat(CL_R, CL_UNORM_INT8);
    cl::Image2D inputImageY(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgformat, width, height, 0, yPlane, &err);
    CHECK_CL(err);
    cl::Image2D inputImageUV(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgformat, width, height/2, 0, uvPlane, &err);
    CHECK_CL(err);
    cl::ImageFormat imgformatOut(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D outputImage(context, CL_MEM_WRITE_ONLY, imgformatOut, width, height, 0, NULL, &err);
    CHECK_CL(err);

    kernelCsc.setArg(0, inputImageY);
    kernelCsc.setArg(1, outputImage);
    kernelCsc.setArg(2, inputImageUV);

    cl::NDRange lws;    // NullRange by default.
    err = commandQueue.finish();
    CHECK_CL(err); 
    for (int iter = 0; iter < 100; iter++)
    {
        err = commandQueue.enqueueNDRangeKernel(
            kernelCsc,
            cl::NullRange,
            cl::NDRange{ size_t(width),  size_t(height) },
            lws);
        CHECK_CL(err);
    }

    commandQueue.finish();
    if (bDump)
    {
        char* outBuffer = new char[height * width * 4];
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { width, height, 1 };

        size_t image_row_pitch = width;
        size_t image_slice_pitch = width * height;
        err = commandQueue.enqueueReadImage(outputImage, CL_TRUE,
            { 0, 0, 0 }, { size_t(width), size_t(height), 1 }, 0, 0, outBuffer, 0, 0);
        DumpData(outBuffer, width * height * 4, "out2_3840x2160.rgba");

    }

    return 0;
}