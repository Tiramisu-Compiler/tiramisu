#include "generated_dot.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#define DATATYPE double


int dot_ref(const int n,
	  const double * const x,
	  const double * const y,
          double * const result)
{
  *result = 0.0;
   double loc_res = 0.0;
  
  #pragma omp parallel for reduction(+: loc_res)
  for (int i=0; i<n; i++)
	loc_res += x[i]*y[i];

  (*result) = loc_res;

  return(0);
}

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

    Halide::Buffer<int> b_SIZES(1);
    b_SIZES(0) = SIZE;
    Halide::Buffer<double> b_x_buf(SIZE);
    init_buffer(b_x_buf, (double) 1);
    Halide::Buffer<double> b_y_buf(SIZE);
    init_buffer(b_y_buf, (double) 1);
    Halide::Buffer<double> b_res_ref(1);
    init_buffer(b_res_ref, (double) 0);
    Halide::Buffer<double> b_res(1);
    init_buffer(b_res, (double) 0);

    {
        for (int i = 0; i < NB_TESTS; i++)
	{
	    init_buffer(b_res_ref, (double)0);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    if (run_ref == true)
	    	dot_ref(SIZE, b_x_buf.data(), b_y_buf.data(), b_res_ref.data());
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
	}
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(b_res, (double)0);
	    auto start2 = std::chrono::high_resolution_clock::now();
 	    if (run_tiramisu == true)
	    	dot(b_SIZES.raw_buffer(), b_x_buf.raw_buffer(), b_y_buf.raw_buffer(), b_res.raw_buffer());
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "sgemm",
               {"Ref", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
 	if (run_ref == 1 && run_tiramisu == 1)
	{
		compare_buffers("dot", b_res_ref, b_res);
        }

    if (PRINT_OUTPUT)
    {
	std::cout << "Tiramisu " << std::endl;
	print_buffer(b_res);
	std::cout << "Reference " << std::endl;
	print_buffer(b_res_ref);
    }

    return 0;
}
