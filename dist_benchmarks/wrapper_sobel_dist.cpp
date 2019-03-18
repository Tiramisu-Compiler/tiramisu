#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_sobel_dist.h"
#include "Halide.h"
#include "halide_image_io.h"

#define COMPARE_TO_HALIDE
#define REQ MPI_THREAD_MULTIPLE

int main() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Halide::Buffer<float> input = Halide::Buffer<float>(NCOLS, NROWS / NNODES + 2);  // middle ranks require two extra rows. 
    for (int y = 0; y < NROWS/NNODES; y++) {
      for (int x = 0; x < input.width(); x++) {
	input(x,1+y) = rank+x+y;
      }
    }

    // Generate these on each node as well
    Halide::Buffer<float> output(input.width(), NROWS/NNODES);
    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    sobel_dist(input.raw_buffer(), output.raw_buffer());

    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    for (int i=0; i<50; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
	sobel_dist(input.raw_buffer(), output.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
    }

    if (rank == 0) {
      print_time("performance_CPU.csv", "### sobel_dist", {"Tiramisu_dist"}, {median(duration_vector)});
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
