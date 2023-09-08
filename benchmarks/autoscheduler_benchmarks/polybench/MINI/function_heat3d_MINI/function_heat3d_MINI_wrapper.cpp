#include "Halide.h"
#include "function_heat3d_MINI_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *c_A = (double*)malloc(10*10*10*sizeof(double));
	parallel_init_buffer(c_A,10*10*10, (double)19);
	Halide::Buffer<double> A(c_A, 10,10,10);
	
	double *c_B = (double*)malloc(10*10*10*sizeof(double));
	parallel_init_buffer(c_B,10*10*10, (double)5);
	Halide::Buffer<double> B(c_B, 10,10,10);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_heat3d_MINI(A.raw_buffer(), B.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}