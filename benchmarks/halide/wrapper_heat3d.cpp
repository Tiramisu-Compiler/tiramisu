#include "Halide.h"
#include "wrapper_heat3d.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include "../benchmarks.h"

int main(int, char **)
{

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> input(_X,_Y,_Z,"data");
    // Init randomly
    //fill data
    srand((unsigned)time(0));//randomize
    for (int z=0; z<_Z; z++) {
      for (int c = 0; c < _Y; c++) {
          for (int r = 0; r < _X; r++)
                input(r, c, z) = rand()%_BASE;
      }
    }

    Halide::Buffer<float> output1(_X, _Y,_Z,_TIME+1,"output1");
    Halide::Buffer<float> output2(_X, _Y,_Z,_TIME+1,"output2");
    Halide::Buffer<float> output_ref(_X, _Y,_Z,"output_ref");
    Halide::Buffer<float> output_tiramisu(_X, _Y,_Z,"output_tiramisu");
    // Warm up code.
    heat3d_tiramisu(input.raw_buffer(), output1.raw_buffer());


    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3d_tiramisu(input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    
    heat3d_ref(input.raw_buffer(), output2.raw_buffer());
    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        heat3d_ref(input.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "heat3d",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    //copy last elements only
    for(int i=0;i<_X;i++)
        for (int j=0;j<_Y;j++)
            for(int k=0;k<_Z;k++)
                output_ref(i,j,k)=output2(i,j,k,_TIME);
    //
    for(int i=0;i<_X;i++)
        for (int j=0;j<_Y;j++)
            for(int k=0;k<_Z;k++)
                output_tiramisu(i,j,k)=output1(i,j,k,_TIME);

    if (CHECK_CORRECTNESS) compare_buffers_approximately("benchmark_heat3d", output_tiramisu, output_ref);

    return 0;
}
