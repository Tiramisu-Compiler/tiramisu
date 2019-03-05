#include "wrapper_blurxyautodist.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {

#ifdef WITH_MPI

  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

  int rank = tiramisu_MPI_init();

  Halide::Buffer<uint32_t> input(_COLS + 2, _ROWS/10, "input");

  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS+2; c++) {
      input(c,r) = r + c;
    }
  }
  if (rank == 9) {
    uint32_t v = _ROWS/10;
    for (int r = _ROWS/10; r < _ROWS/10 + 2; r++) {
      for (int c = 0; c < _COLS + 2; c++) {
        input(c,r) = v + c;
      }
      v++;
    }
  }

  Halide::Buffer<uint32_t> output(_COLS, _ROWS/10, "output");
  Halide::Buffer<uint32_t> ref(_COLS, _ROWS/10, "ref");
  init_buffer(output, (uint32_t)0);

  blurxyautodist_tiramisu(input.raw_buffer(), output.raw_buffer());
  // Tiramisu
  for (int i=0; i<NB_TESTS; i++)
  {
      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      ticket_tiramisu(output1.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
  }
   MPI_Barrier(MPI_COMM_WORLD);

  if(CHECK_CORRECTNESS){
      for (int r = 0; r < _ROWS/10; r++) {
        for (int c = 0; c < _COLS; c++) {
          ref(c,r) = (input(c,r) + input(c,r+1) + input(c,r+2) + input(c+1, r) + input(c+1, r+1) + input(c+1, r+2) + input(c+2, r)
                      + input(c+2, r+1) + input(c+2, r+2)) / 9;
        }
      }

      compare_buffers("blurxyautodist_" + std::to_string(rank), output, ref);
  }

  if(rank == 0) {
      print_time("performance_CPU.csv", "blurxyautodist",
                 {"Tiramisu", "Halide"},
                 {median(duration_vector_1),0});
  }


  tiramisu_MPI_cleanup();
#endif

  return 0;
}
