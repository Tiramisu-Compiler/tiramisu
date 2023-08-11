#include "Halide.h"
#include "function_ludcmp_SMALL_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(120*120*sizeof(double));
	parallel_init_buffer(b_A, 120*120, (double)19);
	Halide::Buffer<double> buf01(b_A, 120, 120);

	double *b_E = (double*)malloc(120* sizeof(double));
	parallel_init_buffer(b_E, 120, (double)36);
	Halide::Buffer<double> buf04(b_E, 120);

	double *b_F = (double*)malloc(120*sizeof(double));
	parallel_init_buffer(b_F, 120, (double)19);
	Halide::Buffer<double> buf05(b_F, 120);

	double *b_B = (double*)malloc(120*sizeof(double));
	parallel_init_buffer(b_B, 120, (double)19);
	Halide::Buffer<double> buf06(b_B, 120);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_ludcmp_SMALL(buf01.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer(), buf06.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
