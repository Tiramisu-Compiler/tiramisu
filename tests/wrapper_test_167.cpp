#include "wrapper_test_167.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

int main() {
#ifdef WITH_MPI
    int rank = tiramisu_MPI_init();
    Halide::Buffer<int> buffer_in1(50, 50, "buffer_i1");
    Halide::Buffer<int> buffer_out1(50, 50, "buffer_o1");
    Halide::Buffer<int> buffer_in2(100, 50, "buffer_i2");
    Halide::Buffer<int> buffer_out2(100, 50, "buffer_o2");
    
    if (rank == 4 || rank == 5) {
      for (int i = 0; i < 50; i++) {
	for (int j = 0; j < 100; j++) {
	  buffer_in2(j,i) = rank;
	}
      }
    } else {
      for (int i = 0; i < 50; i++) {
	for (int j = 0; j < 50; j++) {
	  buffer_in1(j,i) = rank;
	}
      }
    }

    dist_topo_mapping(buffer_in1.raw_buffer(), buffer_in2.raw_buffer(),
		      buffer_out1.raw_buffer(), buffer_out2.raw_buffer());

    if (rank == 4 || rank == 5) {
      for (int i = 0; i < 50; i++) {
	for (int j = 0; j < 100; j++) {
	  if (buffer_out2(j,i) != rank - 22) {
	    std::cerr << "rank is " << rank << " and got " << buffer_out2(i,j) << " at (" << i 
		      << "," << j << ")" << std::endl;
	    abort();
	  }
	}
      }
    } else {
      for (int i = 0; i < 50; i++) {
	for (int j = 0; j < 50; j++) {
	  if (buffer_out1(j,i) != rank * 17) {
	    std::cerr << "rank is " << rank << " and got " << buffer_out2(i,j) << " at (" << i 
		      << "," << j << ")" << std::endl;
	    abort();
	  }
	}
      }
    }


    tiramisu_MPI_cleanup();
#endif
    return 0;
}
