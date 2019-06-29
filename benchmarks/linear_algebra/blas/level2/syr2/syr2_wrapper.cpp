#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_syr2.o.h"
#include "benchmarks.h"
#include <tiramisu/utils.h>

int syr2_ref(int n, double alpha, Halide::Buffer<double> A, Halide::Buffer<double> x, Halide::Buffer<double> y)
{
    for (int i = 0; i < n; ++i)
       for (int j = 0; j < n; j++)
          A(j , i) +=  alpha * x(i) * y(j) + alpha * x(j) * y(i);  

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

    double alpha = 0.5;
    Halide::Buffer<double> b_alpha(1);
    b_alpha(0) = alpha;
    Halide::Buffer<double> b_X(N), b_Y(N), b_A(N, N), b_A_ref(N, N);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {

            init_buffer(b_A_ref, (double) 1);
	    init_buffer(b_X, (double) 2);
            init_buffer(b_Y, (double) 3);
            auto start = std::chrono::high_resolution_clock::now();

            if (run_ref)
	    	syr2_ref(N, alpha, b_A_ref, b_X, b_Y );

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_A, (double)1);
	    init_buffer(b_X, (double) 2);
	    init_buffer(b_Y, (double)3);
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	syr2(b_A.raw_buffer(), b_X.raw_buffer(), b_Y.raw_buffer(), b_alpha.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "syr2",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("syr2", b_A_ref, b_A);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_A);

        std::cout << "Reference " << std::endl;
        print_buffer(b_A_ref);
    }

    return 0;
}
