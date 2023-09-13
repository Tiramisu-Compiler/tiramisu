#include "Halide.h"
#include "function_bicg_MEDIUM_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(410*390*sizeof(double));
	parallel_init_buffer(b_A, 410*390, (double)19);
	Halide::Buffer<double> buf01(b_A, 410, 390);

	double *b_B = (double*)malloc(390* sizeof(double));
	parallel_init_buffer(b_B, 390, (double)36);
	Halide::Buffer<double> buf02(b_B, 390);

	double *b_C = (double*)malloc(410*sizeof(double));
	parallel_init_buffer(b_C, 410, (double)19);
	Halide::Buffer<double> buf03(b_C, 410);

	double *b_D = (double*)malloc(410* sizeof(double));
	parallel_init_buffer(b_D, 410, (double)36);
	Halide::Buffer<double> buf04(b_D, 410);

	double *b_E = (double*)malloc(390*sizeof(double));
	parallel_init_buffer(b_E, 390, (double)19);
	Halide::Buffer<double> buf05(b_E, 390);



	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_bicg_MEDIUM(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
