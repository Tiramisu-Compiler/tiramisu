#include "Halide.h"
#include "wrapper_heat3ddist.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"
#include "tiramisu/mpi_comm.h"

int main(int, char**) {
    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;
    std::cout << "I'm rank == " << rank << std::endl;
    //start executing tiramisu version
    Halide::Buffer<float> node_input(_X,_Y,_Z/NODES,"data");//buffer specific for each node
    init_buffer(node_input,(float)0);
    srand((unsigned)time(0));
    for (int z=0; z<_Z/NODES; z++) {
      for (int c = 0; c < _Y; c++) {
          for (int r = 0; r < _X; r++)
                node_input(r, c, z) = rand()%10+1+rank; //init data on each node
      }
    }
    Halide::Buffer<float> node_output(_X, _Y, _Z/NODES+2,_TIME+1, "output");
    init_buffer(node_output, (float)0);
    //warm up
    heat3ddist(node_input.raw_buffer(), node_output.raw_buffer());
    // Tiramisu
    for (int i=0; i<1; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3ddist(node_input.raw_buffer(), node_output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //gather all inputs, and outputs
    float in_node[_Z/NODES][_Y][_X];
    float in_global[_Z][_Y][_X]; //change with dynamic allocation and allocate only when rank=0

    //copy node inputs in temporary buff
    for (int z=0; z<_Z/NODES; z++) {
        for (int c = 0; c < _Y; c++) {
            for (int r = 0; r < _X; r++)
                in_node[z][c][r]=node_input(r, c, z);
          }
    }
    MPI_Gather(in_node, _X*_Y*_Z/NODES, MPI_FLOAT,in_global, _X*_Y*_Z/NODES, MPI_FLOAT,0,MPI_COMM_WORLD);
    //if z%NODES!=0 the last node will need to send more than _X*_Y*_Z/NODES a more sophisticated test
    //will be written

    float out_node[_Z/NODES][_Y][_X];
    float out_global[_Z][_Y][_X];//change with dynamic allocation and allocate only when rank=0
    for(int z=1;z<_Z/NODES+1;z++){
        for (int r=0; r<_X; r++){
            for (int c = 0; c < _Y; c++)
                   out_node[z-1][c][r]=node_output(r,c,z,_TIME);
            }
    }
    MPI_Gather(out_node,_X*_Y*_Z/NODES,MPI_FLOAT,out_global,_X*_Y*_Z/NODES,MPI_FLOAT,0,MPI_COMM_WORLD);

    if(rank==0){
        //copy to a halide buffer
        Halide::Buffer<float> input_halide(_X, _Y, _Z,"input_halide");
        Halide::Buffer<float> output_halide(_X, _Y, _Z,_TIME+1,"output_halide");
        for (int z=0; z<_Z; z++) {
            for (int c = 0; c < _Y; c++) {
                for (int r = 0; r < _X; r++)
                    input_halide(r, c, z)= in_global[z][c][r];
            }
        }
        //warm up
        heat3ddist_ref(input_halide.raw_buffer(), output_halide.raw_buffer());

        for (int i = 0; i < 1; i++) {
            auto start2 = std::chrono::high_resolution_clock::now();
            heat3ddist_ref(input_halide.raw_buffer(), output_halide.raw_buffer());
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration2 = end2 - start2;
            duration_vector_2.push_back(duration2);
        }

        print_time("performance_CPU.csv", "Heat3d Dist",
                   {"Tiramisu", "Halide"},
                   {median(duration_vector_1), median(duration_vector_2)});

        if (CHECK_CORRECTNESS) {
            //comparison
            Halide::Buffer<float> output_ref(_X, _Y, _Z,"output_ref");
            Halide::Buffer<float> output_tiramisu(_X, _Y, _Z,"output_tiramisu");
            for (int z=0; z<_Z; z++) {
                for (int c = 0; c < _Y; c++) {
                    for (int r = 0; r < _X; r++)
                            {
                                output_ref(r,c,z) = output_halide(r,c,z,_TIME);
                                output_tiramisu(r,c,z)=out_global[z][c][r];//from gatherd result
                            }
                          }
                    }
            compare_buffers_approximately(" heat3d " , output_tiramisu,output_ref);
        }
        std::cout << "Distributed heat3d passed" << std::endl;
    }

    tiramisu_MPI_cleanup();
    return 0;
}
