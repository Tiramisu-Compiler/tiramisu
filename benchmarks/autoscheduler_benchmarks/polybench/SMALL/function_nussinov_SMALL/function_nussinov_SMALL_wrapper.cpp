#include "Halide.h"
#include "function_nussinov_SMALL_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	int *b_D = (int*)malloc(180*180*sizeof(int));
	parallel_init_buffer(b_D, 180*180, (int)19);
	Halide::Buffer<int> buf03(b_D, 180, 180);

	int *b_B = (int*)malloc(180*sizeof(int));
	parallel_init_buffer(b_B, 180, (int)19);
	Halide::Buffer<int> buf06(b_B, 180);




	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_nussinov_SMALL(buf03.raw_buffer(), buf06.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
