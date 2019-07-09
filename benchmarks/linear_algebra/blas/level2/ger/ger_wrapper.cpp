#include "Halide.h"
#include <tiramisu/tiramisu.h>
#include "generated_ger.o.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

int ger_ref(const int MM, const int NN,const double * A, const double * x,const double * y, const double * alpha, double * result)
	{
	    for(int i = 0; i < MM; i++)
	    {
	        for(int j = 0; j < NN; j++)
		{
		        result[j * MM + i] = alpha[0] * x[i] * y[j] + A[j * MM + i];
	        }
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
  Halide::Buffer<double> b_alpha(1), b_x(MM),b_y(NN), b_A(NN, MM), b_result(NN, MM), b_result_ref(NN, MM);
    init_buffer(b_alpha, (double) 2);
    init_buffer(b_x, (double) 2);
    init_buffer(b_y, (double) 3);
    init_buffer(b_A, (double) 1);
    init_buffer(b_result, (double) 0);
    init_buffer(b_result_ref, (double) 0);

    {
    	        for (int i = 0; i < NB_TESTS; ++i)
    	        {
    	            auto start = std::chrono::high_resolution_clock::now();

    	            if (run_ref)
    		    	ger_ref(MM,NN, b_A.data(), b_x.data(),b_y.data(), b_alpha.data(), b_result_ref.data());

    	            auto end = std::chrono::high_resolution_clock::now();
    	            duration_vector_1.push_back(end - start);
    	        }
    }
    {
       for (int i = 0; i < NB_TESTS; ++i)
       {
           auto start = std::chrono::high_resolution_clock::now();

           if (run_tiramisu)
       ger(b_A.raw_buffer(), b_x.raw_buffer(),b_y.raw_buffer(), b_alpha.raw_buffer(), b_result.raw_buffer());

           auto end = std::chrono::high_resolution_clock::now();
           duration_vector_2.push_back(end - start);
       }
   }
   print_time("performance_cpu.csv", "ger",
          {"Ref", "Tiramisu"},
          {median(duration_vector_1), median(duration_vector_2)});
    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
      	 compare_buffers("ger", b_result, b_result_ref);

     if (PRINT_OUTPUT)
      	 {
      	        std::cout << "Tiramisu " << std::endl;
      	        print_buffer(b_result);

      	        std::cout << "Reference " << std::endl;
      	        print_buffer(b_result_ref);
      	 }
    return 0;
}
