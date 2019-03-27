#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_cvtcolor_dist.h"
#include "Halide.h"
#include "halide_image_io.h"

#define REQ MPI_THREAD_MULTIPLE

int main() {
  int provided = -1;
  MPI_Init_thread(NULL, NULL, REQ, &provided);
  assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int64_t rows_per_rank = NROWS / NUM_MPI_RANKS;
  Halide::Buffer<uint32_t> input(NCOLS, rows_per_rank, 3);
  for (int64_t y = 0; y < rows_per_rank; y++) {
    for (int64_t x = 0; x < input.width(); x++) {
      for (int64_t c = 0; c < 3; c++) {
	input(x,y,c) = rank+x+y+c;
      }
    }
  }

  // Generate these on each node as well
  Halide::Buffer<uint32_t> output(input.width(), rows_per_rank);
  // Run once to get rid of overhead/any extra compilation stuff that needs to happen
  cvtcolor_dist(input.raw_buffer(), output.raw_buffer());

  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  for (int i=0; i<15; i++) {
    if (rank == 0)
      std::cerr << i << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    cvtcolor_dist(input.raw_buffer(), output.raw_buffer());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
  }

  if (rank == 0) {
    print_time("performance_CPU.csv", "### cvtcolor_dist", {"Tiramisu_dist"}, {median(duration_vector)});
  }

  MPI_Barrier(MPI_COMM_WORLD);
    
#ifdef COMPARE_TO_HALIDE
  std::string fname = "/data/scratch/jray/oopsla_2019/tiramisu/build/rank_" + std::to_string(rank) + ".txt";
  std::ofstream out_file;
  out_file.open(fname);
  for (int y = 0; y < output.height(); y++) {
    for (int x = 0; x < output.width(); x++) {	
      out_file << output(x,y) << " ";
    }
  }
  out_file.close();
#endif

  MPI_Finalize();
  
  return 0;
}
