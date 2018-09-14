#include "fusion.h"

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

    Halide::Buffer<uint8_t> f(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> g(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> h(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> k(input.width(), input.height(), input.channels());

    std::cout << "Dimensions : " << std::endl;
    std::cout << "input.extent(0): " << input.extent(0) << std::endl;  // Rows
    std::cout << "input.extent(1): " << input.extent(1) << std::endl;  // Cols
    std::cout << "input.extent(2): " << input.extent(2) << std::endl;  // Colors

    #ifdef __PROFILE_CUDA__
    cudaProfilerStop();
    #endif

    // Warm up
    pencil_fusion(input.extent(0), input.extent(1), input.extent(1), (uint8_t *) input.raw_buffer()->host,
                  (uint8_t *) f.raw_buffer()->host,
                  (uint8_t *) g.raw_buffer()->host,
                  (uint8_t *) h.raw_buffer()->host,
                  (uint8_t *) k.raw_buffer()->host);

    #ifdef __PROFILE_CUDA__
    cudaProfilerStart();
    #endif

    // Tiramisu
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pencil_fusion(input.extent(0), input.extent(1), input.extent(1), (uint8_t *) input.raw_buffer()->host,
                      (uint8_t *) f.raw_buffer()->host,
                      (uint8_t *) g.raw_buffer()->host,
                      (uint8_t *) h.raw_buffer()->host,
                      (uint8_t *) k.raw_buffer()->host);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        duration_vector.push_back(duration);
    }

    std::cout << "time: " << median(duration_vector) << std::endl;

    return 0;
}
