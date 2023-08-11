#include "Halide.h"
#include "function_2mm_MINI_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{
    double *c_b_A = (double*)malloc(22*16*sizeof(double));
	parallel_init_buffer(c_b_A,22*16, (double)2);
	Halide::Buffer<double> b_A(c_b_A, 22,16);

	double *c_b_B = (double*)malloc(18*22*sizeof(double));
	parallel_init_buffer(c_b_B,18*22, (double)19);
	Halide::Buffer<double> b_B(c_b_B, 18,22);

	double *c_b_C = (double*)malloc(24*18*sizeof(double));
	parallel_init_buffer(c_b_C, 24*18, (double)29);
	Halide::Buffer<double> b_C(c_b_C,24,18);

	double *c_b_D = (double*)malloc(24*16*sizeof(double));
	parallel_init_buffer(c_b_D, 24*16, (double)29);
	Halide::Buffer<double> b_D(c_b_D,24,16);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_2mm_MINI(b_A.raw_buffer(),b_B.raw_buffer(),b_C.raw_buffer(),b_D.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}