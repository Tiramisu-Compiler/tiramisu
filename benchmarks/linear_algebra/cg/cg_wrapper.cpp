#include "generated_cg.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"
#include <omp.h>

#define DATATYPE double


void waxpby_ref(const int n,
	       const double alpha,
               const double * const x, 
	       const double beta,
               const double * const y, 
  	       double * const w)
{

  #pragma omp parallel for
  for (int i=0; i<n; i++)
	w[i] = alpha * x[i] + beta * y[i];
}

void cg_ref(const int n,
	   const double alpha,
           const double * const x, 
	   const double beta,
           const double * const y, 
  	   double * const w)
{
 	waxpby_ref(n, alpha, x, beta, y, w);
}

#define nrow SIZE

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    bool run_ref = false;
    bool run_tiramisu = false;

    const char* env_ref = std::getenv("RUN_REF");
    if ((env_ref != NULL) && (env_ref[0] == '1'))
	run_ref = true;
    const char* env_tira = std::getenv("RUN_TIRAMISU");
    if ((env_tira != NULL) && (env_tira[0] == '1'))
	run_tiramisu = true;

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    double alpha = 1;
    double beta = 1;

    Halide::Buffer<double> b_r(nrow);
    init_buffer(b_r, (double) 2);
    Halide::Buffer<double> b_p(nrow);
    init_buffer(b_p, (double) 3);
    Halide::Buffer<double> b_p_ref(nrow);
    init_buffer(b_p_ref, (double) 3);

    Halide::Buffer<double> b_alpha(1);
    init_buffer(b_alpha, (double) alpha);
    Halide::Buffer<double> b_beta(1);
    init_buffer(b_beta, (double) beta);

    Halide::Buffer<double> b_w(nrow);
    init_buffer(b_w, (double) 0);
    Halide::Buffer<double> b_w_ref(nrow);
    init_buffer(b_w_ref, (double) 0);

    {
        for (int i = 0; i < NB_TESTS; i++)
	{
	    init_buffer(b_p_ref, (double)1);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    if (run_ref == true)
	    {
	    	cg_ref(nrow, alpha, b_r.data(), beta, b_p_ref.data(), b_p_ref.data());
	    }
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
	}
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(b_p, (double)1);
	    auto start2 = std::chrono::high_resolution_clock::now();
 	    if (run_tiramisu == true)
	    {
	    	cg(b_r.raw_buffer(), b_beta.raw_buffer(), b_p.raw_buffer());
	    }
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "cg",
               {"Ref", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
 	if (run_ref == 1 && run_tiramisu == 1)
	{
		compare_buffers("cg", b_p_ref, b_p);
        }

    if (PRINT_OUTPUT)
    {
	std::cout << "Tiramisu " << std::endl;
	print_buffer(b_p);
	std::cout << "Reference " << std::endl;
	print_buffer(b_p_ref);
    }

    return 0;
}
