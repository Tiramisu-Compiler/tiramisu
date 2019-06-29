
#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_swap.o.h"
#include "benchmarks.h"
#include <tiramisu/utils.h>

int swap_ref(int n, Halide::Buffer<float> a, Halide::Buffer<float> b)
{
    Halide::Buffer<float> b_temp(N);

    for (int i = 0; i < n; ++i)
    {
      b_temp(i) = a(i);  
      a(i) = b(i); 
      b(i) = b_temp(i);
    }
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


    Halide::Buffer<float> b_b(N), b_a(N), b_a_ref(N), b_b_ref(N);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {

            init_buffer(b_b_ref, (float) 2);
	    init_buffer(b_a_ref, (float) 1);
            auto start = std::chrono::high_resolution_clock::now();

            if (run_ref)
	    	swap_ref(N, b_a_ref, b_b_ref );

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_b, (float) 2);
	    init_buffer(b_a, (float) 1);
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	swap(b_a.raw_buffer(), b_b.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "swap",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("swap", b_a_ref, b_a);
        compare_buffers("swap", b_b_ref, b_b);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_a);
        print_buffer(b_b);

        std::cout << "Reference " << std::endl;
        print_buffer(b_a_ref);
        print_buffer(b_b_ref);
    }

    return 0;
}
