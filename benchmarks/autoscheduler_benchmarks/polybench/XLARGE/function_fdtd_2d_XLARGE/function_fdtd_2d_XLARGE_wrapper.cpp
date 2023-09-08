#include "Halide.h"
#include "function_fdtd_2d_XLARGE_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{



	double *b_C = (double*)malloc(2000*2600* sizeof(double));
	parallel_init_buffer(b_C, 2000*2600, (double)36);
	Halide::Buffer<double> buf02(b_C, 2000, 2600);

	double *b_D = (double*)malloc(2000*2600*sizeof(double));
	parallel_init_buffer(b_D, 2000*2600, (double)19);
	Halide::Buffer<double> buf03(b_D, 2000, 2600);

	double *b_E = (double*)malloc(2000*2600* sizeof(double));
	parallel_init_buffer(b_E, 2000*2600, (double)36);
	Halide::Buffer<double> buf04(b_E, 2000, 2600);



	double *b_B = (double*)malloc(1000*sizeof(double));
	parallel_init_buffer(b_B, 1000, (double)19);
	Halide::Buffer<double> buf06(b_B, 1000);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_fdtd_2d_XLARGE(buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf06.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
