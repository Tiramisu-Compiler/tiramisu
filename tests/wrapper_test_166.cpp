#include "wrapper_test_166.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();

    Halide::Buffer<int> buffer_in(100, 100, "buffer");
    Halide::Buffer<int> buffer_out(100, 100, "buffer");
    
    for (int i = 0; i < 100; i++) {
      for (int j = 0; j < 100; j++) {
	buffer_in(i,j) = rank;
      }
    }

    dist_nonblocking(buffer_in.raw_buffer(), buffer_out.raw_buffer());
    int comp_rank = rank == 0 ? rank : rank - 1;
    for (int i = 0; i < 100; i++) {
      for (int j = 0; j < 100; j++) {
	if (buffer_out(i,j) != comp_rank) {
	  std::cerr << "rank " << rank << " (" << i << "," << j << ") " << buffer_out(i,j) << " should be " << comp_rank << std::endl;
	  abort();
	}
      }
    }

    tiramisu_MPI_cleanup();
#endif
    return 0;
}
