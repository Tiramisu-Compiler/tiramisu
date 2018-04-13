#include "wrapper_heat2d_dist.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#if defined(USE_MPI) || defined(USE_COOP)
#include <mpi.h>
#endif
#if defined(USE_GPU) || defined(USE_COOP)
#include <cuda_runtime.h>
#include
#endif

#define REQ MPI_THREAD_FUNNELED

// TODO (Jess): define how to map ranks to either CPU or GPU (which should go where)
#include <vector>
int main(int, char**)
{
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (NUM_CPU_RANKS != 0) {
      assert(((N-2) * CPU_SPLIT) % 10 == 0);
      assert((((N-2) * CPU_SPLIT) / 10) % NUM_CPU_RANKS == 0);
    }
    if (NUM_GPU_RANKS != 0) {
      assert(((N-2) * GPU_SPLIT) % 10 == 0);
      assert((((N-2) * GPU_SPLIT) / 10) % NUM_GPU_RANKS == 0);
    }
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    // figure out how many rows should be associated with this rank
    int rank_N;
    if (rank < NUM_CPU_RANKS) {
      rank_N = (((N-2) * CPU_SPLIT) / 10) / NUM_CPU_RANKS;
      std::cerr << "Rank " << rank << " is on the CPU and has " << rank_N << " out of " << N << std::endl;
    } else {
#if defined(USE_GPU) || defined(USE_COOP)
      assert(cudaSetDevice(rank % 4) == cudaSuccess && "cudaSetDevice failed.");
#endif
      rank_N = (((N-2) * GPU_SPLIT) / 10) / NUM_GPU_RANKS;
      std::cerr << "Rank " << rank << " is on the GPU and has " << rank_N << " out of " << N << std::endl;
    }
    //    else {
    //      assert(false && "Rank not found.");
    //    }
    
    //    Halide::Buffer<float> input(Halide::Float(32), N, rank_N+2);
    halide_buffer_t input;
    float *_input = (float*)malloc(sizeof(float) * N * (rank_N+2));
    input.host = (uint8_t*)(_input);
    // Init
    std::cerr << "Filling init" << std::endl;
    for (int64_t i = 0; i < rank_N; i++) { 
      for (int64_t j = 0; j < N; j++) {
	_input[(i+1)*N+j] = 1.0f;
      }
    }


    //    Halide::Buffer<float> output1(N, rank_N);
    halide_buffer_t output1;
    output1.host = (uint8_t*)malloc(sizeof(float)*(N+2)*rank_N);

    // Warm up code.
    std::cerr << "Warm up" << std::endl;
    heat2d_dist_tiramisu(&input, &output1);
    // Tiramisu
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<N_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
	MPI_Barrier(MPI_COMM_WORLD);
	heat2d_dist_tiramisu(&input, &output1);
	MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    if (rank == 0) {
      print_time("performance_CPU.csv", "heat2d",
		 {"Tiramisu"},
		 {median(duration_vector_1)});
      std::cerr << std::endl;
    }

    //    if (CHECK_CORRECTNESS)
    //      compare_buffers_approximately("benchmark_heat2d", output1, output2);
    if (CHECK_CORRECTNESS) {
      std::cerr << "Filling full input" << std::endl;
      Halide::Buffer<float> full_input (Halide::Float(32), N, rank_N+2); // contains borders and stuff
      for (int i = 0; i < rank_N; i++) { // this isn't modified after this point
	for (int j = 0; j < N; j++) {
	  ((float*)(full_input.raw_buffer()->host))[(i+1)*N+j] = 1.0f;
	}
      }


      std::cerr << "checking correctness" << std::endl;
      for (int i = 1; i < rank_N-2; i++) {
	for (int j = 1; j < N-2; j++) {
	  float expected = full_input(j,i) * 0.3 + (full_input(j,i-1)+full_input(j,i+1)+full_input(j-1,i)+full_input(j+1,i))*0.4;
	  float got = ((float*)(output1.host))[i*N+j];//(j,i);
	  if (std::fabs(expected - got) > 0.01f) {
	    std::cerr << "Rank " << rank << " expected " << expected << " but got " << got << " at (" << i << ", " << j << ")" << std::endl;
	    assert(false);
	  }
	}
      }
    }

    
    MPI_Finalize();
    
    return 0;
}
