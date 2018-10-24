#include "warpAffine.h"

#include <stdlib.h>
#ifdef __PROFILE_CUDA__
#include <cuda_profiler_api.h>
#endif
#include <cstdlib>
#include <iostream>

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"

int main(int, char**) {
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("../rgb.png");

    Halide::Buffer<float> output1(input.width(), input.height());

    #ifdef __PROFILE_CUDA__
    cudaProfilerStop();
    #endif

    // Warm up
    pencil_affine_linear(input.extent(0), input.extent(1), input.extent(1),
                         (uint8_t *) input.raw_buffer()->host,
                         output1.extent(0), output1.extent(1), output1.extent(1),
                         (float *) output1.raw_buffer()->host,
                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

    std::cout << "Warmup done" << std::endl << std::flush;

    #ifdef __PROFILE_CUDA__
    cudaProfilerStart();
    #endif

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pencil_affine_linear(input.extent(0), input.extent(1), input.extent(1),
                             (uint8_t *) input.raw_buffer()->host,
                             output1.extent(0), output1.extent(1), output1.extent(1),
                             (float *) output1.raw_buffer()->host,
                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        duration_vector.push_back(duration);
    }

    std::cout << "time: " << median(duration_vector) << std::endl;

    return 0;
}
