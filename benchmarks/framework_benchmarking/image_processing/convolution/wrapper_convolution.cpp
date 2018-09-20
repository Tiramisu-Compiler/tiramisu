#include "convolution.h"

#ifdef __PROFILE_CUDA__
#include <cuda_profiler_api.h>
#endif
#include <cstdlib>
#include <iostream>

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"

int main(int, char**)
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("../rgb.png");

    Halide::Buffer<float> kernel(3, 3);
    kernel(0,0) = 0; kernel(0,1) = 1.0f/5; kernel(0,2) = 0;
    kernel(1,0) = 1.0f/5; kernel(1,1) = 1.0f/5; kernel(1,2) = 1.0f/5;
    kernel(2,0) = 0; kernel(2,1) = 1; kernel(2,2) = 0;

    // Small size discrepancy with Halide benchmark:
    Halide::Buffer<uint8_t> output(input.width(), input.height(), input.channels());

    std::cout << "Dimensions : " << std::endl;
    std::cout << "input.extent(0): " << input.extent(0) << std::endl;  // Rows
    std::cout << "input.extent(1): " << input.extent(1) << std::endl;  // Cols
    std::cout << "input.extent(2): " << input.extent(2) << std::endl;  // Colors

    #ifdef __PROFILE_CUDA__
    cudaProfilerStop();
    #endif

    // Warm up
    pencil_convolution(input.extent(0), input.extent(1), input.extent(1),
                       (uint8_t *) input.raw_buffer()->host,
                       (float *) kernel.raw_buffer()->host,
                       (uint8_t *) output.raw_buffer()->host);

    #ifdef __PROFILE_CUDA__
    cudaProfilerStart();
    #endif

    // Tiramisu
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pencil_convolution(input.extent(0), input.extent(1), input.extent(1),
                           (uint8_t *) input.raw_buffer()->host,
                           (float *) kernel.raw_buffer()->host,
                           (uint8_t *) output.raw_buffer()->host);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        duration_vector.push_back(duration);
    }

    std::cout << "time: " << median(duration_vector) << std::endl;

    return 0;
}
