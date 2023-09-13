#include "Halide.h"
#include "function_correlation_LARGE_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_data = (double*)malloc(1400*1200*sizeof(double));
	parallel_init_buffer(b_data,1400*1200, (double)19);
	Halide::Buffer<double> buf01(b_data, 1400,1200);

	double *b_corr = (double*)malloc(1200*1200* sizeof(double));
	parallel_init_buffer(b_corr, 1200*1200, (double)36);
	Halide::Buffer<double> buf02(b_corr, 1200, 1200);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_correlation_LARGE(buf01.raw_buffer(), buf02.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
