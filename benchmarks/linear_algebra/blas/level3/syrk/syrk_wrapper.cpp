#include "generated_syrk.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>

#include <iostream>
#include "benchmarks.h"

#define N_DIM N
#define K_DIM K

int syrk_ref(
	const int NN,
	const int KK,
	const double * A,
	const double * C,
	const double * alpha,
	const double * beta,
		  double * result)
{
	int tmp;
	for(int i = 0; i<NN; i++)
	{
		for(int j = 0; j<=i; j++)
		{
			tmp=0;
			for(int k = 0; k<KK; k++)
				tmp += A[i * KK + k] * A[j * KK + k];
			result[i * NN + j] = alpha[0] * tmp + beta[0] * C[i * NN + j];
		}
	}
  	//Copy the lower part of the matrix into the upper part
  	for(int i=0; i<NN; i++)
	{
		for(int j = i+1; j<NN; j++)
		{
			result[i * NN + j] = result[j * NN + i];
		}
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
	SIZES(0) = N_DIM;
	SIZES(1) = K_DIM;
	
	Halide::Buffer<double> b_A(N_DIM, K_DIM);
	init_buffer(b_A, (double) 1);
	
	Halide::Buffer<double> b_C(N_DIM, N_DIM);
	init_buffer(b_C, (double) 1);
	
	Halide::Buffer<double> b_alpha(1),b_beta(1);
	init_buffer(b_alpha, (double) 3);
	init_buffer(b_beta, (double) 2);
	
	Halide::Buffer<double> b_result(N_DIM, N_DIM), b_result_ref(N_DIM, N_DIM);
	init_buffer(b_result, (double) 0);
	/**
		We have
	    - A a N_DIM by K_DIM matrix of ones
	    - C a N_DIM by N_DIM matrix of ones
	    - alpha, beta scalars respectively equal to 3 and 2
	*/
	
	//REFERENCE
	{
		for (int i = 0; i < NB_TESTS; i++)
		{
			init_buffer(b_result_ref, (double)0);
			auto start1 = std::chrono::high_resolution_clock::now();
			if (run_ref)
				syrk_ref(SIZES(0), SIZES(1), b_A.data(), b_C.data(), b_alpha.data(), b_beta.data(), b_result_ref.data());
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
				syrk(SIZES.raw_buffer(), b_A.raw_buffer(), b_C.raw_buffer(), b_alpha.raw_buffer(), b_beta.raw_buffer(), b_result.raw_buffer());
			auto end = std::chrono::high_resolution_clock::now();
			duration_vector_2.push_back(end - start);
		}
	}
	
	print_time("performance_CPU.csv", "syrk",
				{"Ref", "Tiramisu"},
				{median(duration_vector_1), median(duration_vector_2)});
	
	if (PRINT_OUTPUT)
	{
		if (run_tiramisu)
		{
			std::cout << "Tiramisu " << std::endl;
			print_buffer(b_result);
		}
		if (run_ref)
		{
			std::cout << "Reference " << std::endl;
			print_buffer(b_result_ref);
		}
	}
	
	if (run_ref && run_tiramisu)
		compare_buffers("syrk", b_result, b_result_ref);
	
	return 0;
}