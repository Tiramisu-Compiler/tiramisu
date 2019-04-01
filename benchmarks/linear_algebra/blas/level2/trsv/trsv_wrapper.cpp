#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_trsv.o.h"
#include "benchmarks.h"

#define MAT_N N

int trsv_ref(int n, double* const A, double* const b, double* X)
{
    double forward;
    
    for (int i = 0; i < n; ++i) {
        forward = 0;
	for (int j = 0; j < i; ++j)
	    forward += A[i*n + j]*X[j];

        X[i] = (b[i] - forward) / A[i*n + i];
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
    
    Halide::Buffer<int> SIZES(1);
    SIZES(0) = MAT_N;

    Halide::Buffer<double> b_A(MAT_N, MAT_N);
    Halide::Buffer<double> b_b(MAT_N);
    Halide::Buffer<double> b_X(MAT_N), b_X_ref(MAT_N);

    /* 
     * The example here is of the form :
     * | 1 0 0 ... 0 | |X1|   |1|
     * | 1 2 0 ... 0 | |X2|   |2|
     * | 1 2 3 ... 0 | |X3| = |3|
     * | ........... | |..|   |.|
     * | 1 2 3 ... N | |XN|   |N|
     * 
     * The solutions are of the form : Xk = 1/k
     */
    init_buffer(b_A, (double)0);
    for (int i = 0; i < MAT_N; ++i)
        for (int j = 0; j <= i; ++j)
	    b_A(j, i) = j+1;

    for (int i = 0; i < MAT_N; ++i)
        b_b(i) = i+1;
    
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
      
            if (run_ref)
	        trsv_ref(MAT_N, b_A.data(), b_b.data(), b_X_ref.data());
      
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	        trsv(SIZES.raw_buffer(), b_A.raw_buffer(), b_b.raw_buffer(), b_X.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "trsv",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("trsv", b_X_ref, b_X);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_X);

        std::cout << "Reference " << std::endl;
        print_buffer(b_X_ref);
    }
  
    return 0;
}
