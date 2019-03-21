#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_gaussian_dist.h"
#include "Halide.h"
#include "halide_image_io.h"
#include <cmath>

#define COMPARE_TO_HALIDE
#define REQ MPI_THREAD_MULTIPLE

int main() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Halide::Buffer<float> input = Halide::Buffer<float>(NCOLS, NROWS / NNODES, 3);
    Halide::Buffer<float> gaussian_x = Halide::Buffer<float>(NCOLS, NROWS / NNODES + 4, 3);
    for (int y = 0; y < NROWS/NNODES; y++) {
      for (int x = 0; x < input.width(); x++) {
	for (int c = 0; c < input.channels(); c++) {
	  input(x, y, c) = (x+y+c)*rank;
	}
      }
    }

    // Generate these on each node as well
    Halide::Buffer<float> output(input.width(), NROWS/NNODES, 3);
    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    gaussian_dist(input.raw_buffer(), gaussian_x.raw_buffer(), output.raw_buffer());

    /*    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    for (int i=0; i<50; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
	gaussian_dist(input.raw_buffer(), gaussian_x.raw_buffer(), output.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
	}

    if (rank == 0) {
      print_time("performance_CPU.csv", "### gaussian_dist", {"Tiramisu_dist"}, {median(duration_vector)});
      }*/

    MPI_Barrier(MPI_COMM_WORLD);
    
#ifdef COMPARE_TO_HALIDE
    std::string fname = "/data/scratch/jray/oopsla_2019/tiramisu/build/rank_" + std::to_string(rank) + ".txt";
    std::ofstream out_file;
    out_file.open(fname);
    for (int y = 0; y < output.height(); y++) {
      for (int x = 0; x < output.width(); x++) {	
	for (int c = 0; c < output.channels(); c++) {	
	  out_file << floor(output(x,y,c)) << " ";
	}
      }
    }
    out_file.close();
#endif

    MPI_Finalize();

    return 0;
}
