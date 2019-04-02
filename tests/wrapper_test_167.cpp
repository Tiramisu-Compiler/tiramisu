#include "wrapper_test_167.h"
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

    dist_topo_mapping(buffer_in.raw_buffer(), buffer_out.raw_buffer());
    tiramisu_MPI_cleanup();
#endif
    return 0;
}
