#include "wrapper_edgegpu.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>

#include <tiramisu/utils.h>

#define N 8192
#define M 8192
#define NB_TESTS 100
#define CHECK_CORRECTNESS 1

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;

    Halide::Buffer<uint8_t> Img(N, M, 3);
    Halide::Buffer<uint8_t> output1(N-2, M-2, 3);
    Halide::Buffer<uint8_t> output2(N-4, M-4, 3);

    init_buffer(Img, (uint8_t) 0);
    init_buffer(output1, (uint8_t) 0);
    init_buffer(output2, (uint8_t) 0);

    cudaProfilerStop();

    //Warm up
    edge_tiramisu(Img.raw_buffer(), output1.raw_buffer());

    cudaProfilerStart();

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        edge_tiramisu(Img.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    std::cout << "Tiramisu run time: " << median(duration_vector_1) << std::endl;

    // There is no reference function since cyclic operations cannot be implemented in Halide.

    return 0;
}
