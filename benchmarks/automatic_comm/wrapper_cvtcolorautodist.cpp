#include "wrapper_cvtcolorautodist.h"
#include "../benchmarks.h"
#include <tiramisu/mpi_comm.h>
#include "Halide.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**) {
    int rank = tiramisu_MPI_init();

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input(3, _COLS, _ROWS/NODES );
    Halide::Buffer<uint8_t> output_ref(_COLS, _ROWS/NODES);
    Halide::Buffer<uint8_t> output(_COLS, _ROWS/NODES);

    init_buffer(input, (uint8_t) 0);
    init_buffer(output, (uint8_t) 0);
    init_buffer(output_ref, (uint8_t) 0);

    for (int chan = 0; chan < 3; chan++) {
        for (int c = 0; c < _COLS; c++) {
            for (int r = 0; r < _ROWS/NODES; r++) {
                input(chan, c, r) = 3;
            }
        }
    }

    cvtcolorautodist_tiramisu(input.raw_buffer(), output.raw_buffer());

    for (int i = 0; i < NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        cvtcolorautodist_tiramisu(input.raw_buffer(), output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    cvtcolorautodist_ref(input.raw_buffer(), output.raw_buffer());

    for (int i = 0; i < NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        cvtcolorautodist_ref(input.raw_buffer(), output_ref.raw_buffer());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end - start;
        duration_vector_2.push_back(duration);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    compare_buffers_approximately("CvtColor rank "+std::to_string(rank) , output, output_ref);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        print_time("performance_CPU.csv", "cvtcolor",
                   {"Tiramisu auto", "Tiramisu man"},
                   {median(duration_vector_1), median(duration_vector_2)});

        std::cout << "Distributed cvtcolor passed" << std::endl;
    }

    tiramisu_MPI_cleanup();
    return 0;
}
