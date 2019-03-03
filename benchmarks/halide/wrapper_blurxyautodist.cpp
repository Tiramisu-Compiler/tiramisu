#include "wrapper_blurxyautodist.h"
#include "../benchmarks.h"
#include "../../include/tiramisu/mpi_comm.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

#define NB_TESTS 30
#define CHECK_CORRECTNESS 1
#define NODES 10

int main(int, char**) {

    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    // Allocate buffers for each rank and fill as appropriate
    Halide::Buffer<uint32_t> input(_COLS, _ROWS/NODES + 2, "input");
    Halide::Buffer<uint32_t> output(_COLS, _ROWS/NODES, "output");

    for (int r = 0; r < _ROWS/NODES; r++) {
      // Repeat data at the edge of the columns
      for (int c = 0; c < _COLS; c++) {
        input(c,r) = r + c; // could fill this with anything
      }
    }
    if (rank == NODES -1) {
      // Since the last node doesn't receive any data, we need to repeat the data at the edge of the rows
      uint32_t v = _ROWS/NODES;
      for (int r = _ROWS/NODES; r < _ROWS/NODES + 2; r++) {
        for (int c = 0; c < _COLS; c++) {
          // mimic transferred data
          input(c,r) = v + c;
        }
        v++;
      }
    }

    // Warm up code.
    blurxyautodist_tiramisu(input.raw_buffer(), output.raw_buffer());
    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        blurxyautodist_tiramisu(input.raw_buffer(), output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(CHECK_CORRECTNESS) {

        Halide::Buffer<uint32_t> ref(_COLS, _ROWS/NODES, "ref");
        for (int r = 0; r < _ROWS/NODES; r++) {
          for (int c = 0; c < _COLS; c++) {
            ref(c,r) = (input(c,r) + input(c,r+1) + input(c,r+2) + input(c+1, r) + input(c+1, r+1) + input(c+1, r+2) + input(c+2, r)
                        + input(c+2, r+1) + input(c+2, r+2)) / 9;
          }
        }

        // Compare the results for this rank
        compare_buffers("Distributed blurxy " + std::to_string(rank), output, ref);
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "cvtcolor",
                   {"Tiramisu", "Node"},
                   {median(duration_vector_1), 0});

        std::cout << "Distributed blurxy passed" << std::endl;
    }

    tiramisu_MPI_cleanup();
    return 0;
}
