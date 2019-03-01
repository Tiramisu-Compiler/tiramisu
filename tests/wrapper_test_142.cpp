#include "wrapper_test_142.h"
#include "Halide.h"
#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI

  int rank = tiramisu_MPI_init();

  Halide::Buffer<uint32_t> img(_COLS + 2, _ROWS/10 + 2, "img");

  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS+2; c++) {
      img(c,r) = r + c;
    }
  }
  if (rank == 9) {
    uint32_t v = _ROWS/10;
    for (int r = _ROWS/10; r < _ROWS/10 + 2; r++) {
      for (int c = 0; c < _COLS + 2; c++) {
        img(c,r) = v + c;
      }
      v++;
    }
  }

  Halide::Buffer<uint32_t> output(_COLS, _ROWS/10, "output");
  Halide::Buffer<uint32_t> refrence(_COLS, _ROWS/10, "refrence");

  init_buffer(output, (uint32_t)0);

  MPI_Barrier(MPI_COMM_WORLD);
  boxblur(img.raw_buffer(), output.raw_buffer());
  MPI_Barrier(MPI_COMM_WORLD);

  for (int r = 0; r < _ROWS/10; r++) {
    for (int c = 0; c < _COLS; c++) {
      refrence(c,r) = (img(c,r) + img(c,r+1) + img(c,r+2) + img(c+1, r) + img(c+1, r+1) + img(c+1, r+2) + img(c+2, r)
                  + img(c+2, r+1) + img(c+2, r+2)) / 9;
    }
  }

  compare_buffers(std::string(TEST_NAME_STR) + std::to_string(rank), output, refrence);

  tiramisu_MPI_cleanup();

#endif
  return 0;
}
