#include "wrapper_blurdist.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"

int main() {

#ifdef WITH_MPI

  int rank = tiramisu_MPI_init();
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

  Halide::Buffer<uint32_t> input(_COLS + 2, _ROWS/10+2, "input");
  Halide::Buffer<uint32_t> output(_COLS, _ROWS/10, "output");
  Halide::Buffer<uint32_t> ref(_COLS, _ROWS/10, "ref");


  init_buffer(input, (uint32_t)0);
  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS+2; c++) {
      input(c,r) = r + c;
    }
  }
  init_buffer(output, (uint32_t)0);

  MPI_Barrier(MPI_COMM_WORLD);
  blurdist_tiramisu(input.raw_buffer(), output.raw_buffer());

  MPI_Barrier(MPI_COMM_WORLD);
  init_buffer(input, (uint32_t)0);
  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS+2; c++) {
      input(c,r) = r + c;
    }
  }
  init_buffer(ref, (uint32_t)0);

  for (int i=0; i<NB_TESTS; i++)
  {
      MPI_Barrier(MPI_COMM_WORLD);
      init_buffer(input, (uint32_t)0);
      for (int r = 0; r < _ROWS/10; r++) {
        for (int c = 0; c < _COLS+2; c++) {
          input(c,r) = r + c;
        }
      }
      init_buffer(output, (uint32_t)0);

      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      blurdist_tiramisu(input.raw_buffer(), output.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  blurdist_ref(input.raw_buffer(), ref.raw_buffer());
  for (int i=0; i<NB_TESTS; i++)
  {
      MPI_Barrier(MPI_COMM_WORLD);
      init_buffer(input, (uint32_t)0);
      for (int r = 0; r < _ROWS/10; r++) {
        for (int c = 0; c < _COLS+2; c++) {
          input(c,r) = r + c;
        }
      }
      init_buffer(ref, (uint32_t)0);

      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      blurdist_ref(input.raw_buffer(), ref.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_2.push_back(duration1);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if(CHECK_CORRECTNESS){
      compare_buffers("blurdist_" + std::to_string(rank), output, ref);
  }

  if(rank == 0) {
      print_time("performance_CPU.csv", "blurdist",
                 {"auto", "man"},
                 {median(duration_vector_1), median(duration_vector_2)});
  }

  tiramisu_MPI_cleanup();
#endif

  return 0;
}
