#include "generated_sger.o.h"

#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <tiramisu/utils.h>

#include <iostream>
#include "benchmarks.h"

#define M_DIM M
#define N_DIM N

int sger_ref(int n, int m, double alpha, double * A, double* x, double * y)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            A[i * m + j] +=  alpha * x[i] * y[j];

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
  
    double alpha = 2.0;
  
    Halide::Buffer<double> b_alpha(1);
    b_alpha(0) = alpha;
  
    // b_A and b_A_ref needs to be initialized in each iteration test
    Halide::Buffer<double> b_A(N_DIM, M_DIM), b_A_ref(N_DIM, M_DIM);
    
    Halide::Buffer<double> b_X(N_DIM);
    init_buffer(b_X, (double) 2);

    Halide::Buffer<double> b_Y(M_DIM);
    init_buffer(b_Y, (double) 3);
    
    /**
    * We have :
    * 
    * alpha : equals 2
    * b_X : size N_DIM vector with all values set to 2
    * b_Y : size M_DIM vector with all values set to 3
    * b_A : N_DIM by M_DIM matrix with all values set to 1 (initialized in each loop)
    * b_A_ref : N_DIM by M_DIM matrix with all values set to 1 (initialized in each loop)
    *
    * The result must be a N_DIM by M_DIM with all values equal to 13 
    */

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            // b_A_ref initialized with 1
            init_buffer(b_A_ref, (double) 1);
            auto start = std::chrono::high_resolution_clock::now();
      
            if (run_ref)
	    	sger_ref(N_DIM, M_DIM, alpha, b_A_ref.data(), b_X.data(), b_Y.data() );
      
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            // b_A initialized with 1
            init_buffer(b_A, (double) 1);
			
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	sger(b_A.raw_buffer(), b_X.raw_buffer(), b_Y.raw_buffer(), b_alpha.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "sger",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("sger", b_A_ref, b_A);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_A);

        std::cout << "Reference " << std::endl;
        print_buffer(b_A_ref);
    }
  
    return 0;
}
