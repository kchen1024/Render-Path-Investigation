#include <CL/sycl.hpp>
#include <iostream>
#include <random>

class ConvolutionKernel {
public:
    ConvolutionKernel(sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::global_buffer> input,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output1,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output2,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output3,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output4,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output5,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output6,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output7,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output8,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output9,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output10,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output11,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output12,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output13,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output14,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output15,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output16,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output17,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output18,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output19,
        sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output20,
        sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> weights,
        sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> biases)
        : input(input), output1(output1), output2(output2), output3(output3), output4(output4), output5(output5), 
        output6(output6), output7(output7), output8(output8), output9(output9), output10(output10),
        output11(output11), output12(output12), output13(output13), output14(output14), output15(output15),
        output16(output16), output17(output17), output18(output18), output19(output19), output20(output20),
        weights(weights), biases(biases) {}

    void operator()(sycl::nd_item<2> item) {
        int row = item.get_global_id(0);
        int col = item.get_global_id(1);
        float result1 = biases[0];
        float result2 = biases[1];
        float result3 = biases[2];
        float result4 = biases[3];
        float result5 = biases[4];
        float result6 = biases[5];
        float result7 = biases[6];
        float result8 = biases[7];
        float result9 = biases[8];
        float result10 = biases[9];
        float result11 = biases[10];
        float result12 = biases[11];
        float result13 = biases[12];
        float result14 = biases[13];
        float result15 = biases[14];
        float result16 = biases[15];
        float result17 = biases[16];
        float result18 = biases[17];
        float result19 = biases[18];
        float result20 = biases[19];

        // Zero Padding
        if (row < output1.get_range()[0] && col < output1.get_range()[1]) {
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    int inputRow = row + i;
                    int inputCol = col + j;
                    if (inputRow < input.get_range()[0] && inputCol < input.get_range()[1]) {
                        result1 += input[inputRow][inputCol] * weights[0 * 9 + i * 3 + j];
                        result2 += input[inputRow][inputCol] * weights[1 * 9 + i * 3 + j];
                        result3 += input[inputRow][inputCol] * weights[2 * 9 + i * 3 + j];
                        result4 += input[inputRow][inputCol] * weights[3 * 9 + i * 3 + j];
                        result5 += input[inputRow][inputCol] * weights[4 * 9 + i * 3 + j];
                        result6 += input[inputRow][inputCol] * weights[5 * 9 + i * 3 + j];
                        result7 += input[inputRow][inputCol] * weights[6 * 9 + i * 3 + j];
                        result8 += input[inputRow][inputCol] * weights[7 * 9 + i * 3 + j];
                        result9 += input[inputRow][inputCol] * weights[8 * 9 + i * 3 + j];
                        result10 += input[inputRow][inputCol] * weights[9 * 9 + i * 3 + j];
                        result11 += input[inputRow][inputCol] * weights[10 * 9 + i * 3 + j];
                        result12 += input[inputRow][inputCol] * weights[11 * 9 + i * 3 + j];
                        result13 += input[inputRow][inputCol] * weights[12 * 9 + i * 3 + j];
                        result14 += input[inputRow][inputCol] * weights[13 * 9 + i * 3 + j];
                        result15 += input[inputRow][inputCol] * weights[14 * 9 + i * 3 + j];
                        result16 += input[inputRow][inputCol] * weights[15 * 9 + i * 3 + j];
                        result17 += input[inputRow][inputCol] * weights[16 * 9 + i * 3 + j];
                        result18 += input[inputRow][inputCol] * weights[17 * 9 + i * 3 + j];
                        result19 += input[inputRow][inputCol] * weights[18 * 9 + i * 3 + j];
                        result20 += input[inputRow][inputCol] * weights[19 * 9 + i * 3 + j];
                    }
                }
            }
        }

        output1[row][col] = (result1 > 0) ? result1 : result1 * 0.1f; // PReLU activation
        output2[row][col] = (result1 > 0) ? result2 : result2 * 0.1f; // PReLU activation
        output3[row][col] = (result1 > 0) ? result3 : result3 * 0.1f; // PReLU activation
        output4[row][col] = (result1 > 0) ? result4 : result4 * 0.1f; // PReLU activation
        output5[row][col] = (result1 > 0) ? result5 : result5 * 0.1f; // PReLU activation
        output6[row][col] = (result1 > 0) ? result6 : result6 * 0.1f; // PReLU activation
        output7[row][col] = (result1 > 0) ? result7 : result7 * 0.1f; // PReLU activation
        output8[row][col] = (result1 > 0) ? result8 : result8 * 0.1f; // PReLU activation
        output9[row][col] = (result1 > 0) ? result9 : result9 * 0.1f; // PReLU activation
        output10[row][col] = (result1 > 0) ? result10 : result10 * 0.1f; // PReLU activation
        output11[row][col] = (result1 > 0) ? result11 : result11 * 0.1f; // PReLU activation
        output12[row][col] = (result1 > 0) ? result12 : result12 * 0.1f; // PReLU activation
        output13[row][col] = (result1 > 0) ? result13 : result13 * 0.1f; // PReLU activation
        output14[row][col] = (result1 > 0) ? result14 : result14 * 0.1f; // PReLU activation
        output15[row][col] = (result1 > 0) ? result15 : result15 * 0.1f; // PReLU activation
        output16[row][col] = (result1 > 0) ? result16 : result16 * 0.1f; // PReLU activation
        output17[row][col] = (result1 > 0) ? result17 : result17 * 0.1f; // PReLU activation
        output18[row][col] = (result1 > 0) ? result18 : result18 * 0.1f; // PReLU activation
        output19[row][col] = (result1 > 0) ? result19 : result19 * 0.1f; // PReLU activation
        output20[row][col] = (result1 > 0) ? result20 : result20 * 0.1f; // PReLU activation
    }

private:
    sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::global_buffer> input;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output1;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output2;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output3;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output4;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output5;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output6;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output7;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output8;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output9;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output10;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output11;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output12;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output13;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output14;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output15;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output16;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output17;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output18;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output19;
    sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> output20;
    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> weights;
    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> biases;
};

int main() {
    constexpr size_t inputChannels = 1;
    constexpr size_t outputChannels = 20;
    constexpr size_t inputHeight = 1080;
    constexpr size_t inputWidth = 1920;
    constexpr size_t outputHeight = inputHeight;
    constexpr size_t outputWidth = inputWidth;

    std::vector<float> input(inputChannels * inputHeight * inputWidth);
    std::vector<float> weights(outputChannels * inputChannels * 5 * 5);
    std::vector<float> biases(outputChannels);
    std::vector<float> output1(outputHeight * outputWidth);
    std::vector<float> output2(outputHeight * outputWidth);
    std::vector<float> output3(outputHeight * outputWidth);
    std::vector<float> output4(outputHeight * outputWidth);
    std::vector<float> output5(outputHeight * outputWidth);
    std::vector<float> output6(outputHeight * outputWidth);
    std::vector<float> output7(outputHeight * outputWidth);
    std::vector<float> output8(outputHeight * outputWidth);
    std::vector<float> output9(outputHeight * outputWidth);
    std::vector<float> output10(outputHeight * outputWidth);
    std::vector<float> output11(outputHeight * outputWidth);
    std::vector<float> output12(outputHeight * outputWidth);
    std::vector<float> output13(outputHeight * outputWidth);
    std::vector<float> output14(outputHeight * outputWidth);
    std::vector<float> output15(outputHeight * outputWidth);
    std::vector<float> output16(outputHeight * outputWidth);
    std::vector<float> output17(outputHeight * outputWidth);
    std::vector<float> output18(outputHeight * outputWidth);
    std::vector<float> output19(outputHeight * outputWidth);
    std::vector<float> output20(outputHeight * outputWidth);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 255.0f);
    for (int i = 0; i < inputChannels * inputHeight * inputWidth; ++i) {
        input[i] = dis(gen);
    }

    for (int i = 0; i < outputChannels * inputChannels * 3 * 3; ++i) {
        weights[i] = dis(gen);
    }

    for (int i = 0; i < outputChannels; ++i) {
        biases[i] = dis(gen);
    }

    std::fill(output1.begin(), output1.end(), 0.0f);
    std::fill(output2.begin(), output2.end(), 0.0f);
    std::fill(output3.begin(), output3.end(), 0.0f);
    std::fill(output4.begin(), output4.end(), 0.0f);
    std::fill(output5.begin(), output5.end(), 0.0f);
    std::fill(output6.begin(), output6.end(), 0.0f);
    std::fill(output7.begin(), output7.end(), 0.0f);
    std::fill(output8.begin(), output8.end(), 0.0f);
    std::fill(output9.begin(), output9.end(), 0.0f);
    std::fill(output10.begin(), output10.end(), 0.0f);
    std::fill(output11.begin(), output11.end(), 0.0f);
    std::fill(output12.begin(), output12.end(), 0.0f);
    std::fill(output13.begin(), output13.end(), 0.0f);
    std::fill(output14.begin(), output14.end(), 0.0f);
    std::fill(output15.begin(), output15.end(), 0.0f);
    std::fill(output16.begin(), output16.end(), 0.0f);
    std::fill(output17.begin(), output17.end(), 0.0f);
    std::fill(output18.begin(), output18.end(), 0.0f);
    std::fill(output19.begin(), output19.end(), 0.0f);
    std::fill(output20.begin(), output20.end(), 0.0f);

    // Create SYCL queue
    sycl::queue queue(sycl::gpu_selector{});

    // Create buffers
    sycl::buffer<float, 2> inputBuffer(input.data(), sycl::range<2>(inputHeight, inputWidth));
    sycl::buffer<float, 2> outputBuffer1(output1.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer2(output2.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer3(output3.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer4(output4.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer5(output5.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer6(output6.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer7(output7.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer8(output8.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer9(output9.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer10(output10.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer11(output11.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer12(output12.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer13(output13.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer14(output14.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer15(output15.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer16(output16.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer17(output17.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer18(output18.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer19(output19.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 2> outputBuffer20(output20.data(), sycl::range<2>(outputHeight, outputWidth));
    sycl::buffer<float, 1> weightsBuffer(weights.data(), sycl::range<1>(outputChannels * inputChannels * 3 * 3));
    sycl::buffer<float, 1> biasesBuffer(biases.data(), sycl::range<1>(outputChannels));

    sycl::range<2> local_size(16, 64);

    // Calculate global size based on output dimensions and local size
    size_t globalSizeX = (outputChannels + local_size[0] - 1) / local_size[0]; // Round up to ensure all output channels are covered
    size_t globalSizeY = (outputHeight + local_size[1] - 1) / local_size[1]; // Round up to ensure all output heights are covered
    sycl::range<2> global_size(globalSizeX * local_size[0], globalSizeY * local_size[1]);
    for (int i = 0; i < 1; i++)
    {
        // Submit kernel to the queue
        queue.submit([&](sycl::handler& cgh) {
            auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
            auto outputAccessor1 = outputBuffer1.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor2 = outputBuffer2.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor3 = outputBuffer3.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor4 = outputBuffer4.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor5 = outputBuffer5.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor6 = outputBuffer6.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor7 = outputBuffer7.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor8 = outputBuffer8.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor9 = outputBuffer9.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor10 = outputBuffer10.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor11 = outputBuffer11.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor12 = outputBuffer12.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor13 = outputBuffer13.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor14 = outputBuffer14.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor15 = outputBuffer15.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor16 = outputBuffer16.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor17 = outputBuffer17.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor18 = outputBuffer18.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor19 = outputBuffer19.get_access<sycl::access::mode::write>(cgh);
            auto outputAccessor20 = outputBuffer20.get_access<sycl::access::mode::write>(cgh);
            auto weightsAccessor = weightsBuffer.get_access<sycl::access::mode::read>(cgh);
            auto biasesAccessor = biasesBuffer.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class Convolution>(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
                ConvolutionKernel(inputAccessor, outputAccessor1, outputAccessor2, outputAccessor3, outputAccessor4, outputAccessor5,
                outputAccessor6, outputAccessor7, outputAccessor8, outputAccessor9, outputAccessor10,
                outputAccessor11, outputAccessor12, outputAccessor13, outputAccessor14, outputAccessor15,
                outputAccessor16, outputAccessor17, outputAccessor18, outputAccessor19, outputAccessor20, weightsAccessor, biasesAccessor)(item);
                });
            });
    }

    // Wait for the computation to finish
    queue.wait();

    return 0;
}
