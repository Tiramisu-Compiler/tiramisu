#include "Halide.h"
#include "function_gramschmidt_MINI_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(20*30*sizeof(double));
	parallel_init_buffer(b_A, 20*30, (double)19);
	Halide::Buffer<double> buf01(b_A, 20,30);

	double *b_C = (double*)malloc(20*30* sizeof(double));
	parallel_init_buffer(b_C, 20*30, (double)36);
	Halide::Buffer<double> buf02(b_C, 20, 30);

	double *b_D = (double*)malloc(30*30*sizeof(double));
	parallel_init_buffer(b_D, 30*30, (double)19);
	Halide::Buffer<double> buf03(b_D, 30, 30);



	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_gramschmidt_MINI(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
