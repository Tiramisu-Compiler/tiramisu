#include "Halide.h"
#include "function_3mm_MEDIUM_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(180*200*sizeof(double));
	parallel_init_buffer(b_A, 180*200, (double)19);
	Halide::Buffer<double> buf01(b_A, 180, 200);

	double *b_B = (double*)malloc(200*190* sizeof(double));
	parallel_init_buffer(b_B, 200*190, (double)36);
	Halide::Buffer<double> buf02(b_B, 200,190);

	double *b_C = (double*)malloc(190*220*sizeof(double));
	parallel_init_buffer(b_C, 190*220, (double)19);
	Halide::Buffer<double> buf03(b_C, 190, 220);

	double *b_D = (double*)malloc(220*210* sizeof(double));
	parallel_init_buffer(b_D, 220*210, (double)36);
	Halide::Buffer<double> buf04(b_D, 220, 210);

	double *b_E = (double*)malloc(180*210*sizeof(double));
	parallel_init_buffer(b_E, 180*210, (double)19);
	Halide::Buffer<double> buf05(b_E, 180, 210);

	
	



	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_3mm_MEDIUM(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
