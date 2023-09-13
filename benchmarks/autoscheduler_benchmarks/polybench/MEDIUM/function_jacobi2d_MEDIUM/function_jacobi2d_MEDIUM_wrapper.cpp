#include "Halide.h"
#include "function_jacobi2d_MEDIUM_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{
    double *c_b_A = (double*)malloc(250*250* sizeof(double));
	parallel_init_buffer(c_b_A, 250*250, (double)61);
	Halide::Buffer<double> b_A(c_b_A, 250,250);

	double *c_b_B = (double*)malloc(250*250* sizeof(double));
	parallel_init_buffer(c_b_B, 250*250, (double)61);
	Halide::Buffer<double> b_B(c_b_B, 250,250);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_jacobi2d_MEDIUM(b_A.raw_buffer(), b_B.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}