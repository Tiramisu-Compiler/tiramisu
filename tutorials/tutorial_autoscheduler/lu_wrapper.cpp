#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "build/generated_lu.o.h"
#include "polybench-tiramisu.h"
#include "lu.h"
#include <tiramisu/utils.h>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{
    Halide::Buffer<double> b_A(N,N);
	
	init_array(b_A);
	

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		lu(b_A.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}

