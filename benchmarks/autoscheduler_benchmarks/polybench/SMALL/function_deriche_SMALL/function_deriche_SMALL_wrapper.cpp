#include "Halide.h"
#include "function_deriche_SMALL_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	float *b_A = (float*)malloc(192*128*sizeof(float));
	parallel_init_buffer(b_A, 192*128, (float)19);
	Halide::Buffer<float> buf01(b_A, 192, 128);

	float *b_x = (float*)malloc(192*128* sizeof(float));
	parallel_init_buffer(b_x, 192*128, (float)36);
	Halide::Buffer<float> buf02(b_x, 192, 128);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_deriche_SMALL(buf01.raw_buffer(), buf02.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
