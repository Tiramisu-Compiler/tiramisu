#include "Halide.h"
#include "function_2mm_MEDIUM_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{
    double *c_b_A = (double*)malloc(210*180*sizeof(double));
	parallel_init_buffer(c_b_A,210*180, (double)2);
	Halide::Buffer<double> b_A(c_b_A, 210,180);

	double *c_b_B = (double*)malloc(190*210*sizeof(double));
	parallel_init_buffer(c_b_B,190*210, (double)19);
	Halide::Buffer<double> b_B(c_b_B, 190,210);

	double *c_b_C = (double*)malloc(220*190*sizeof(double));
	parallel_init_buffer(c_b_C, 220*190, (double)29);
	Halide::Buffer<double> b_C(c_b_C,220,190);

	double *c_b_D = (double*)malloc(220*180*sizeof(double));
	parallel_init_buffer(c_b_D, 220*180, (double)29);
	Halide::Buffer<double> b_D(c_b_D,220,180);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_2mm_MEDIUM(b_A.raw_buffer(),b_B.raw_buffer(),b_C.raw_buffer(),b_D.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}