#include "Halide.h"
#include "function_doitgen_XLARGE_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(250*220*270*sizeof(double));
	parallel_init_buffer(b_A, 250*220*270, (double)19);
	Halide::Buffer<double> buf01(b_A, 250, 220, 270);

	double *b_x = (double*)malloc(270*270* sizeof(double));
	parallel_init_buffer(b_x, 270*270, (double)36);
	Halide::Buffer<double> buf02(b_x, 270, 270);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_doitgen_XLARGE(buf01.raw_buffer(), buf02.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}