#include "wrapper_test_98.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>


int main(int, char **)
{
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();

    Halide::Buffer<int> input({100,100});
    Halide::Buffer<int> output({100,100});
    init_buffer(input, 2);

    Halide::Buffer<int> ref_output({100,100});
    for (int x = 0; x < 100; x++) {
        for (int y = 0; y < 100; y++) {
            ref_output(x,y) = input(x,y);
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int x = 0; x < 100; x++) {
            for (int y = 0; y < 100; y++) {
                ref_output(x,y) *= 2;
            }
        }
    }
    dist_comp_only(input.raw_buffer(), output.raw_buffer());
    compare_buffers(TEST_NAME_STR, output, ref_output);

    tiramisu_MPI_cleanup();
#endif
    return 0;
}