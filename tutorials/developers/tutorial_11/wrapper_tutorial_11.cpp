#include "wrapper_tutorial_11.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    // The rank here is the unique ID of the currently executing process
    int rank = tiramisu_MPI_init();

    // Allocate buffers for each rank and fill as appropriate
    Halide::Buffer<int> input({3, _COLS, _ROWS/10}, "input");
    Halide::Buffer<int> output(_COLS, _ROWS/10, "output");
    init_buffer(input, rank);

    // Synchronize. Really only necessary for timing, which we aren't doing here,
    // but it's a good habit to get into
    MPI_Barrier(MPI_COMM_WORLD);
    cvtcolor(input.raw_buffer(), output.raw_buffer());
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute the reference for this particular rank
    Halide::Buffer<int> ref_output(100, 100, "ref");
    for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
        ref_output(c, r) = ((input(0, c, r) * 4899 + input(1, c, r) * 9617 + input(2, c, r) * 1868) + (1 << 13)) >> 13;
      }
    }

    // Compare the results for this rank
    compare_buffers("tutorial_11_rank_" + std::to_string(rank), output, ref_output);

    // Need to cleanup MPI appropriately
    tiramisu_MPI_cleanup();
#endif
    return 0;
}
