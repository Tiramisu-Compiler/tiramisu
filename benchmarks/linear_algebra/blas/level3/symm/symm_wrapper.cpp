
#include "Halide.h"
#include <tiramisu/utils.h>

#include <iostream>
#include "benchmarks.h"



int symm_ref( const double * A, const double * B, const double * C, const double * alpha, const double * beta, double * result)
{
	int tmp;
	for(int i = 0; i<N; i++){
		for(int j = 0; j<M; j++){
			tmp=0;
			for(int k = 0; k<j-1; k++){
				C[k][j] += alpha * A[k][i] * B[i][j];
	    			temp += B[k][j] * A[k][i];
			}
	  	result[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * tmp;
		}
	}
	
	return 0;
}

int main(int argc, char** argv) {

	std::vector<std::chrono::duration<double, std::milli>> duration_tiramisu;
	std::vector<std::chrono::duration<double, std::milli>> duration_ref;

	bool run_ref = false, run_tiramisu = false;

   	const char* env_ref = std::getenv("RUN_REF");
    	if (env_ref != NULL && env_ref[0] == '1')
        	run_ref = true;

    	const char* env_tiramisu = std::getenv("RUN_TIRAMISU");
    	if (env_tiramisu != NULL && env_tiramisu[0] == '1')
		run_tiramisu = true;

	Halide::Buffer<int> SIZES(2);
	SIZES(0) = N;
	SIZES(1) = M;
	
	Halide::Buffer<double> b_A(N, N);
	init_buffer(b_A, (double) 1);

	Halide::Buffer<double> b_B(N, M);
	init_buffer(b_B, (double) 2);
	
	Halide::Buffer<double> b_C(N, M);
	init_buffer(b_C, (double) 1);

	Halide::Buffer<double> b_alpha(1),b_beta(1);
	init_buffer(b_alpha, (double) 0.5);
	init_buffer(b_beta, (double) 2);

	Halide::Buffer<double> b_result(N, M), b_result_ref(N, M);
	init_buffer(b_result, (double) 0);

	/* With these values the result must be N by M matrix with all values = 5.0 */ 
	
	// Testing C reference 
	{
		for (int i = 0; i < NB_TESTS; ++i)
		{
		    auto start = std::chrono::high_resolution_clock::now();
	      
		    if (run_ref)
			symm_ref(b_A.data(), b_B.data(), b_C.data(), b_alpha.data(), b_beta.data(), b_result_ref.data());
	      
		    auto end = std::chrono::high_resolution_clock::now();
		    duration_ref.push_back(end - start);
		}
	}

	//Testing Tiramisu generated code : 
	{
		for (int i = 0; i < NB_TESTS; ++i)
		{
			auto start = std::chrono::high_resolution_clock::now();
			if (run_tiramisu)
				symm(SIZES.raw_buffer(), b_A.raw_buffer(), b_B.raw_buffer(),  b_C.raw_buffer(), b_alpha.raw_buffer(), b_beta.raw_buffer(), b_result.raw_buffer());
			auto end = std::chrono::high_resolution_clock::now();
			duration_tiramisu.push_back(end - start);
		}
	}
	
	print_time("performance_CPU.csv", "symm",
				{ "Ref", "Tiramisu"},
				{ median(duration_ref), median(duration_tiramisu)});


	return 0; 

}


	

