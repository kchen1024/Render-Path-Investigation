#include <CL/sycl.hpp>
#include <iostream>
#include <random>

class ConvolutionKernel {
public:
    ConvolutionKernel(sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::global_buffer> input,
        sycl::accessor<float, 3, sycl::access::mode::write, sycl::access::target::global_buffer> output,
        sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> weights,
        sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> biases)
        : input(input), output(output), weights(weights), biases(biases) {}

    void operator()(sycl::nd_item<3> item) {
        int ch = item.get_global_id(0);
        int row = item.get_global_id(1);
        int col = item.get_global_id(2);
        float result = biases[ch];

        // Zero Padding
        if (row < output.get_range()[1] && col < output.get_range()[2]) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int inputRow = row + i;
                    int inputCol = col + j;
                    if (inputRow < input.get_range()[0] && inputCol < input.get_range()[1]) {
                        result += input[inputRow][inputCol] * weights[ch * 9 + i * 3 + j];
                    }
                }
            }
        }

        output[ch][row][col] = (result > 0) ? result : result * 0.1f; // PReLU activation
    }

private:
    sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::global_buffer> input;
    sycl::accessor<float, 3, sycl::access::mode::write, sycl::access::target::global_buffer> output;
    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> weights;
    sycl::accessor<float, 1, sycl::access::mode::read, sycl::access::target::global_buffer> biases;
};

int main() {
    constexpr size_t inputChannels = 1;
    constexpr size_t outputChannels = 25;
    constexpr size_t inputHeight = 360;
    constexpr size_t inputWidth = 640;
    constexpr size_t outputHeight = inputHeight;
    constexpr size_t outputWidth = inputWidth;

    std::vector<float> input(inputChannels * inputHeight * inputWidth);
    std::vector<float> weights(outputChannels * inputChannels * 3 * 3);
    std::vector<float> biases(outputChannels);
    std::vector<float> output(outputChannels * outputHeight * outputWidth);

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

    std::fill(output.begin(), output.end(), 0.0f);
    // Create SYCL queue
    sycl::queue queue(sycl::gpu_selector{});

    // Create buffers
    sycl::buffer<float, 2> inputBuffer(input.data(), sycl::range<2>(inputHeight, inputWidth));
    sycl::buffer<float, 3> outputBuffer(output.data(), sycl::range<3>(outputChannels, outputHeight, outputWidth));
    sycl::buffer<float, 1> weightsBuffer(weights.data(), sycl::range<1>(outputChannels * inputChannels * 3 * 3));
    sycl::buffer<float, 1> biasesBuffer(biases.data(), sycl::range<1>(outputChannels));

    //sycl::range<3> global_size(outputChannels, outputHeight, outputWidth);
    //sycl::range<3> local_size(1, 270, 64);

    sycl::range<3> local_size(1, 16, 64);

    // Calculate global size based on output dimensions and local size
    size_t globalSizeX = (outputChannels + local_size[0] - 1) / local_size[0]; // Round up to ensure all output channels are covered
    size_t globalSizeY = (outputHeight + local_size[1] - 1) / local_size[1]; // Round up to ensure all output heights are covered
    size_t globalSizeZ = (outputWidth + local_size[2] - 1) / local_size[2]; // Round up to ensure all output widths are covered

    sycl::range<3> global_size(globalSizeX * local_size[0], globalSizeY * local_size[1], globalSizeZ * local_size[2]);

    for (int i = 0; i < 1; i++)
    {
        // Submit kernel to the queue
        queue.submit([&](sycl::handler& cgh) {
            auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
            auto outputAccessor = outputBuffer.get_access<sycl::access::mode::write>(cgh);
            auto weightsAccessor = weightsBuffer.get_access<sycl::access::mode::read>(cgh);
            auto biasesAccessor = biasesBuffer.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class Convolution>(sycl::nd_range<3>(global_size, local_size), [=](sycl::nd_item<3> item) {
                ConvolutionKernel(inputAccessor, outputAccessor, weightsAccessor, biasesAccessor)(item);
                });
            });
    }

    // Wait for the computation to finish
    queue.wait();

    return 0;
}
