#include "wrapper_tutorial_11.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();
    // For ease, make sure that we can evenly split the data across 10 ranks
    assert(_ROWS % 10 == 0);
    Halide::Buffer<int> input({3, _COLS, _ROWS/10}, "input");
    Halide::Buffer<int> output(_COLS, _ROWS/10, "output");
    init_buffer(input, rank);
    cvtcolor(input.raw_buffer(), output.raw_buffer());
    MPI_Barrier(MPI_COMM_WORLD);
    // Compute the reference
    Halide::Buffer<int> ref_output(100, 100, "ref");
    for (int r = 0; r < 100; r++) {
      for (int c = 0; c < 100; c++) {
        ref_output(c, r) = ((input(0, c, r) * 4899 + input(1, c, r) * 9617 + input(2, c, r) * 1868) + (1 << 13)) >> 13;
      }
    }
    compare_buffers("CVTCOLOR", output, ref_output);

    tiramisu_MPI_cleanup();
#endif
    return 0;
}
