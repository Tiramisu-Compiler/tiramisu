#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_copy.o.h"
#include "benchmarks.h"
#include <tiramisu/utils.h>

int copy_ref(int n,
             Halide::Buffer<float> a,
             Halide::Buffer<float> x)
{
    for (int i = 0; i < n; ++i)
          a(i) =  x(i);  
    return 0;
}


int main(int argc, char** argv)
{    
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1, duration_vector_2;
    bool run_ref = false, run_tiramisu = false;

    const char* env_ref = std::getenv("RUN_REF");
    if (env_ref != NULL && env_ref[0] == '1')
        run_ref = true;

    const char* env_tiramisu = std::getenv("RUN_TIRAMISU");
    if (env_tiramisu != NULL && env_tiramisu[0] == '1')
        run_tiramisu = true;

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    Halide::Buffer<float> b_x(N), b_a(N), b_a_ref(N);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    //REFERENCE
    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_x, (float) i);
            auto start = std::chrono::high_resolution_clock::now();
            if (run_ref)
	    	copy_ref(N, b_a_ref, b_x );
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    //TIRAMISU
    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_x, (float) i);
            auto start = std::chrono::high_resolution_clock::now();
            if (run_tiramisu)
	    	copy(b_a.raw_buffer(), b_x.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }
    print_time("performance_cpu.csv", "copy",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("copy", b_a_ref, b_a);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_a);
        std::cout << "Reference " << std::endl;
        print_buffer(b_a_ref);
    }

    return 0;
}
