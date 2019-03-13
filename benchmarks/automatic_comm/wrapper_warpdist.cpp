#include "wrapper_warpdist.h"
#include "../benchmarks.h"
#include "Halide.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <tiramisu/mpi_comm.h>

int main(int, char**)
{
    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input(_N1/NODES, _N0);

    Halide::Buffer<float> output1(input.width(), input.height()/NODES);
    Halide::Buffer<float> output2(input.width(), input.height()/NODES);
    Halide::Buffer<int> SIZES(2);

    SIZES(0) = input.height()/NODES;
    SIZES(1) = input.height()/NODES

    // Warm up
    warpdist_tiramisu(SIZES.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
    //
    // // Tiramisu
    // for (int i=0; i<NB_TESTS; i++)
    // {
    //     auto start1 = std::chrono::high_resolution_clock::now();
    //     warpdist_tiramisu(SIZES.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
    //     auto end1 = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double,std::milli> duration1 = end1 - start1;
    //     duration_vector_1.push_back(duration1);
    // }
    //
    // // Reference
    // warpdist_ref(input.raw_buffer(), output2.raw_buffer());
    // for (int i=0; i<NB_TESTS; i++)
    // {
    //     auto start2 = std::chrono::high_resolution_clock::now();
    //     warpdist_ref(input.raw_buffer(), output2.raw_buffer());
    //     auto end2 = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    //     duration_vector_2.push_back(duration2);
    // }

    // print_time("performance_CPU.csv", "warpdist",
    //            {"Tiramisu", "Halide"},
    //            {median(duration_vector_1), median(duration_vector_2)});
    //
    // if (CHECK_CORRECTNESS)
	// compare_buffers_approximately("benchmark_warpdist", output1, output2);

    tiramisu_MPI_cleanup();

    return 0;
}
