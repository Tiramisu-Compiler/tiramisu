#include "wrapper_test_99.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();

    Halide::Buffer<int> buffer(100, 100, "buffer");
    Halide::Buffer<int> ref(100, 100, "ref");

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            buffer(j,i) = rank;
            ref(j,i) = rank;
        }
    }
    // Set the first row of rank to be updated
    int prev_rank = (rank == 0) ? 9 : rank - 1; 
    for (int j = 0; j < 100; j++) {
      ref(j,0) = prev_rank;
    }
    dist_comm_only_block(buffer.raw_buffer());
    MPI_Barrier(MPI_COMM_WORLD);
    compare_buffers(TEST_NAME_STR, buffer, ref);
    MPI_Barrier(MPI_COMM_WORLD);

    tiramisu_MPI_cleanup();
#endif
    return 0;
}
