#include "wrapper_edgedist.h"
#include "Halide.h"
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>

#include <tiramisu/utils.h>


#define NB_TESTS 10
#define CHECK_CORRECTNESS 1

int main(int, char**)
{
    int rank  = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<int32_t> Img1(3, _COLS, _ROWS/NODES + 2);
    Halide::Buffer<int32_t> Img2(3, _COLS, _ROWS/NODES + 2);
    Halide::Buffer<int32_t> output1(3, _COLS, _ROWS/NODES + 2);
    Halide::Buffer<int32_t> output2(3, _COLS, _ROWS/NODES + 2);


    init_buffer(output1, (int32_t) 0);
    init_buffer(Img1, (int32_t) 0);
    for(int i = 0; i < _ROWS/NODES; i++)
    {
        for(int j = 0; j < _COLS; j++)
        {
            for(int k = 0 ; k < 3; k++) Img1(k,j,i) = 1 + rank;
        }

    }
    edgedist_tiramisu(Img1.raw_buffer(), output1.raw_buffer());
    //Synchro
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i<NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        init_buffer(output1, (int32_t) 0);
        init_buffer(Img1, (int32_t) 0);
        for(int i = 0; i < _ROWS/NODES; i++)
        {
            for(int j = 0; j < _COLS; j++)
            {
                for(int k = 0 ; k < 3; k++) Img1(k,j,i) = 1 + rank;
            }

        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        edgedist_tiramisu(Img1.raw_buffer(), output1.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    init_buffer(output2, (int32_t) 0);
    init_buffer(Img2, (int32_t) 0);
    for(int i = 0; i < _ROWS/NODES; i++)
    {
        for(int j = 0; j < _COLS; j++)
        {
            for(int k = 0 ; k < 3; k++) Img2(k,j,i) = 1 + rank;
        }

    }
    edgedist_ref(Img2.raw_buffer(), output2.raw_buffer());
    for (int i = 0; i<NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        init_buffer(output2, (int32_t) 0);
        init_buffer(Img2, (int32_t) 0);
        for(int i = 0; i < _ROWS/NODES; i++)
        {
            for(int j = 0; j < _COLS; j++)
            {
                for(int k = 0 ; k < 3; k++) Img2(k,j,i) = 1 + rank;
            }

        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        edgedist_ref(Img2.raw_buffer(), output2.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    compare_buffers_approximately("edgedist rank " + std::to_string(rank), Img1, Img2);


    tiramisu_MPI_cleanup();

    return 0;
}
