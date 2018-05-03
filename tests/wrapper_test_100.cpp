#include "wrapper_test_100.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();

    Halide::Buffer<int> buffer({100, 100});
    Halide::Buffer<int> ref({100,100});

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            buffer(j,i) = i * 100 + j;
            ref(j,i) = i * 100 + j;
        }
    }

    dist_comm_only_nonblock(buffer.raw_buffer());
    compare_buffers(TEST_NAME_STR, buffer, ref);

    tiramisu_MPI_cleanup();
#endif
    return 0;
}