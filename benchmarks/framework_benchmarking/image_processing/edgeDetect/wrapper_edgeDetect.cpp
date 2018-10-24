#include "edgeDetect.h"

#ifdef __PROFILE_CUDA__
#include <cuda_profiler_api.h>
#endif
#include <cstdlib>
#include <iostream>

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"

#define NN 8192
#define MM 8192

int main(int, char**) {
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;

    Halide::Buffer<uint8_t> input(NN, MM, 3);
    Halide::Buffer<uint8_t> output(NN, MM, 3);
    Halide::Buffer<uint8_t> temp(NN, MM, 3);

    std::cout << "Dimensions : " << std::endl;
    std::cout << "input.extent(0): " << input.extent(0) << std::endl;  // Rows
    std::cout << "input.extent(1): " << input.extent(1) << std::endl;  // Cols
    std::cout << "input.extent(2): " << input.extent(2) << std::endl;  // Colors

    #ifdef __PROFILE_CUDA__
    cudaProfilerStop();
    #endif

    // Warm up
    pencil_edge(input.extent(0), input.extent(1), input.extent(1),
                (uint8_t *) input.raw_buffer()->host,
                (uint8_t *) output.raw_buffer()->host,
                (uint8_t *) temp.raw_buffer()->host);

    #ifdef __PROFILE_CUDA__
    cudaProfilerStart();
    #endif

    // Tiramisu
    for (int i = 0; i < 100; i++) {
        auto start1 = std::chrono::high_resolution_clock::now();
        pencil_edge(input.extent(0), input.extent(1), input.extent(1),
                    (uint8_t *) input.raw_buffer()->host,
                    (uint8_t *) output.raw_buffer()->host,
                    (uint8_t *) temp.raw_buffer()->host);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration1 = end1 - start1;
        duration_vector.push_back(duration1);
    }

    std::cout << "time: " << median(duration_vector) << std::endl;

    return 0;
}
