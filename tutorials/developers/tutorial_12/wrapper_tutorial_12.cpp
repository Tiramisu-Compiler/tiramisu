#include "wrapper_tutorial_12.h"
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
  Halide::Buffer<uint32_t> input(_COLS + 2, _ROWS/10 + 2, "input");
  
  for (int r = 0; r < _ROWS/10; r++) {
    // Repeat data at the edge of the columns
    for (int c = 0; c < _COLS+2; c++) {
      input(c,r) = r + c; // could fill this with anything
    }
  }
  if (rank == 9) {
    // Since the last node doesn't receive any data, we need to repeat the data at the edge of the rows
    uint32_t v = _ROWS/10;
    for (int r = _ROWS/10; r < _ROWS/10 + 2; r++) {
      for (int c = 0; c < _COLS + 2; c++) {
        // mimic transferred data
        input(c,r) = v + c;
      }
      v++;
    }    
  }
  
  Halide::Buffer<uint32_t> output(_COLS, _ROWS/10, "output");
  Halide::Buffer<uint32_t> ref(_COLS, _ROWS/10, "ref");
  init_buffer(output, (uint32_t)0);

  // Synchronize. Really only necessary for timing, which we aren't doing here,
  // but it's a good habit to get into
  MPI_Barrier(MPI_COMM_WORLD);
  blurxy(input.raw_buffer(), output.raw_buffer());
  MPI_Barrier(MPI_COMM_WORLD);

  // Compute the reference for this particular rank
  if (rank != 9) {
    uint32_t v = 0;
    for (int r = _ROWS/10; r < _ROWS/10 + 2; r++) {
      for (int c = 0; c < _COLS + 2; c++) {
        // mimic transferred data
        input(c,r) = v + c;
      }
      v++;
    }
  }
  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS; c++) {
      ref(c,r) = (input(c,r) + input(c,r+1) + input(c,r+2) + input(c+1, r) + input(c+1, r+1) + input(c+1, r+2) + input(c+2, r)
                  + input(c+2, r+1) + input(c+2, r+2)) / 9;
    }
  }
  
  // Compare the results for this rank
  compare_buffers("tutorial_12_rank_" + std::to_string(rank), output, ref);

  // Need to cleanup MPI appropriately
  tiramisu_MPI_cleanup();
#endif

  return 0;
}
