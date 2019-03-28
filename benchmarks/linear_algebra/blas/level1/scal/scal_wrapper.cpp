#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_scal.o.h"
#include "benchmarks.h"

#define nrow SIZE

int scal_ref(int n, double alpha, double* X)
{
    for (int i = 0; i < n; ++i)
    	X[i] = alpha * X[i];

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
  
    double alpha = 2.5;
  
    Halide::Buffer<int> SIZES(1);
    SIZES(0) = nrow;
  
    Halide::Buffer<double> b_alpha(1);
    b_alpha(0) = alpha;
  
    Halide::Buffer<double> b_X(nrow), b_X_ref(nrow);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_X_ref, (double)1);
            auto start = std::chrono::high_resolution_clock::now();
      
            if (run_ref)
	    	scal_ref(nrow, alpha, b_X_ref.data());
      
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_X, (double)1);
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	scal(SIZES.raw_buffer(), b_alpha.raw_buffer(), b_X.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "scal",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("scal", b_X_ref, b_X);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_X);

        std::cout << "Reference " << std::endl;
        print_buffer(b_X_ref);
    }
  
    return 0;
}
