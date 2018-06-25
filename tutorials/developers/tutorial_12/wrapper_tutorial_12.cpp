#include "wrapper_tutorial_12.h"

// This tutorial will only be run if tiramisu is compiled with MPI support.
#ifdef WITH_MPI
#include <mpi.h>
#endif

int main() {

#ifdef WITH_MPI
  // The rank here is the unique ID of the currently executing process
  int rank = tiramisu_MPI_init();

  // Allocate buffers for each rank and fill as appropriate
  Halide::Buffer<uint8_t> input(NUM_COLS + 2, NUM_ROWS/NUM_NODES + 2, "input");
  
  for (int r = 0; r < NUM_ROWS/NUM_NODES; r++) {
    // Repeat data at the edge of the columns
    for (int c = 0; c < NUM_COLS+2; c++) {
      input(c,r) = r + c; // could fill this with anything
    }
  }
  if (rank == NUM_NODES - 1) {
    // Since the last node doesn't receive any data, we need to repeat the data at the edge of the rows
    uint8_t v = NUM_ROWS/NUM_NODES;
    for (int r = NUM_ROWS/NUM_NODES; r < NUM_ROWS/NUM_NODES + 2; r++) {
      for (int c = 0; c < NUM_COLS; c++) {
        // mimic transferred data data
        input(c,r) = v + c;
      }
      v++;
    }    
  }
  
  Halide::Buffer<uint8_t> output(NUM_COLS, NUM_ROWS/NUM_NODES, "output");
  Halide::Buffer<uint8_t> ref(NUM_COLS, NUM_ROWS/NUM_NODES, "ref");
  init_buffer(output_buf0, (uint8_t)0);

  // Synchronize. Really only necessary for timing, which we aren't doing here,
  // but it's a good habit to get into
  MPI_Barrier(MPI_COMM_WORLD);
  blurxy(input.raw_buffer(), output.raw_buffer());
  MPI_Barrier(MPI_COMM_WORLD);

  // Compute the reference for this particular rank
  if (rank != NUM_NODES - 1) {
    uint8_t v = 0;
    for (int r = NUM_ROWS/NUM_NODES; r < NUM_ROWS/NUM_NODES + 2; r++) {
      for (int c = 0; c < NUM_COLS; c++) {
        // mimic transferred data data
        input(c,r) = v + c;
      }
      v++;
    }
  }
  for (int r = 0; r < NUM_ROWS/NUM_NODES; r++) {
    for (int c = 0; c < NUM_COLS; c++) {
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
