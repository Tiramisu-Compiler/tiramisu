#include "wrapper_test_101.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();

    Halide::Buffer<int> input(100, rank == 0 ? 1000 : 100, "input");
    Halide::Buffer<int> output(100, rank == 0 ? 1000 : 100, "output");
    if (rank == 0) {
        init_buffer(input, 2);
    }
    dist_reduction(input.raw_buffer(), output.raw_buffer());
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        Halide::Buffer<int> ref_output(100, 1000, "ref");
        for (int x = 0; x < 100; x++) {
            for (int y = 0; y < 1000; y++) {
                ref_output(x, y) = input(x, y);
            }
        }
        for (int i = 0; i < 10; i++) {
            for (int x = 0; x < 100; x++) {
                for (int y = 0; y < 1000; y++) {
                    ref_output(x, y) *= 2;
                }
            }
        }
        compare_buffers(TEST_NAME_STR, output, ref_output);
    }

    tiramisu_MPI_cleanup();
#endif
    return 0;
}
