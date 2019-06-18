
#include "generated_gemv.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>

#include <iostream>
#include "benchmarks.h"

#define M_DIM 2000
#define N_DIM 1000

int gemv_ref(
    const int MM,const int NN,
    const double * A,
    const double * x,
    const double * y,
    const double * alpha,
    const double * beta,
    double * result)
{
  int i=0,j=0;
  int tmp=0;
  for(i=0;i<MM; i++){
    tmp=0;
    for(j=0;j<NN;j++){
      tmp+= A[i*NN+j]*x[j];
    }
    tmp*=alpha[0];
    result[i]= tmp + beta[0]*y[i];
  }
  return 0;
}

int main(int argc, char** argv)
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

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


    Halide::Buffer<int> SIZES(2);
    SIZES(0) = M_DIM;
    SIZES(1) = N_DIM;

    Halide::Buffer<double> b_A(M_DIM, N_DIM);
    init_buffer(b_A, (double) 1);

    Halide::Buffer<double> b_y(M_DIM);
    init_buffer(b_y, (double) 1);

    Halide::Buffer<double> b_x(N_DIM);
    init_buffer(b_x, (double) 1);
    Halide::Buffer<double> b_alpha(1),b_beta(1);
    init_buffer(b_alpha, (double) 1);
    init_buffer(b_beta, (double) 1);

    Halide::Buffer<double> b_result(M_DIM), b_result_ref(M_DIM);
    init_buffer(b_result, (double) 0);
    /**
      We have
        - A a MxN matrix of ones
        - x a size N Vector of ones
        - y a size M Vector of ones
        - alpha,beta scalars equal to one
    */


    //REFERENCE
    {
        for (int i = 0; i < NB_TESTS; i++)
	{
            init_buffer(b_result_ref, (double)0);
      	    auto start1 = std::chrono::high_resolution_clock::now();
      	    if (run_ref)
      	       gemv_ref(SIZES(0),SIZES(1), b_A.data(), b_x.data(), b_y.data(), b_alpha.data(), b_beta.data(), b_result_ref.data());
      	    auto end1 = std::chrono::high_resolution_clock::now();
      	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      	    duration_vector_1.push_back(duration1);
        }
    }


    //TIRAMISU
    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	             gemv(SIZES.raw_buffer(), b_A.raw_buffer(), b_x.raw_buffer(), b_y.raw_buffer(), b_alpha.raw_buffer(), b_beta.raw_buffer(), b_result.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_CPU.csv", "gemv",
               {"Ref", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_2)});


    if (run_ref && run_tiramisu)
        compare_buffers("gemv", b_result_ref, b_result);

    return 0;
}
