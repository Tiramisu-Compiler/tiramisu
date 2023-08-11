#include "Halide.h"
#include "function_doitgen_MINI_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(10*8*12*sizeof(double));
	parallel_init_buffer(b_A, 10*8*12, (double)19);
	Halide::Buffer<double> buf01(b_A, 10, 8, 12);

	double *b_x = (double*)malloc(12*12* sizeof(double));
	parallel_init_buffer(b_x, 12*12, (double)36);
	Halide::Buffer<double> buf02(b_x, 12, 12);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_doitgen_MINI(buf01.raw_buffer(), buf02.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
