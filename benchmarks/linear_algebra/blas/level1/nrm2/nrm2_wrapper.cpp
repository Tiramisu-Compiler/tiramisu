#include "generated_nrm2.o.h"

#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <tiramisu/utils.h>

#include <iostream>
#include "benchmarks.h"
#include <math.h>

#define M_DIM M
#define N_DIM N

int nrm2_ref(int n, double * X, double *nrm)
{   
    double sum = 0;
	
    for (int i = 0; i < n; ++i)
    	 sum += X[i] * X[i]; 
	
    nrm[0] = sqrt(sum);
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
  
    Halide::Buffer<double> b_result(1), b_result_ref(1);
	
    Halide::Buffer<double> b_X(N_DIM);
    init_buffer(b_X, (double) 1);
	
    /**
       X vector is initialized to 1
    **/
	
    {
        for (int i = 0; i < NB_TESTS; ++i)
        {   
            auto start = std::chrono::high_resolution_clock::now();
		
            if (run_ref)
	    	nrm2_ref(N_DIM, b_X.data(), b_result_ref.data() );
		
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {	
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	nrm2(b_X.raw_buffer(), b_result.raw_buffer());
		
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "nrm2",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("nrm2", b_result, b_result_ref);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_result);

        std::cout << "Reference " << std::endl;
        print_buffer(b_result_ref);
    }
  
    return 0;
}
