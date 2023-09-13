#include "Halide.h"
#include "function_gemver_MEDIUM_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(400*400*sizeof(double));
	parallel_init_buffer(b_A,400*400, (double)19);
	Halide::Buffer<double> buf01(b_A, 400, 400);

	double *b_u1 = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_u1, 400, (double)36);
	Halide::Buffer<double> buf02(b_u1, 400);

	double *b_u2 = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_u2, 400, (double)36);
	Halide::Buffer<double> buf03(b_u2, 400);

	double *b_v1 = (double*)malloc(400*sizeof(double));
	parallel_init_buffer(b_v1,400, (double)19);
	Halide::Buffer<double> buf04(b_v1, 400);

	double *b_v2 = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_v2, 400, (double)36);
	Halide::Buffer<double> buf05(b_v2, 400);

	double *b_y = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_y, 400, (double)36);
	Halide::Buffer<double> buf06(b_y, 400);

	double *b_z = (double*)malloc(400*sizeof(double));
	parallel_init_buffer(b_z,400, (double)19);
	Halide::Buffer<double> buf07(b_z, 400);

	double *b_w = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_w, 400, (double)36);
	Halide::Buffer<double> buf08(b_w, 400);

	double *b_A_hat = (double*)malloc(400*400*sizeof(double));
	parallel_init_buffer(b_A_hat,400*400, (double)19);
	Halide::Buffer<double> buf09(b_A_hat, 400, 400);

	double *b_x = (double*)malloc(400* sizeof(double));
	parallel_init_buffer(b_x, 400, (double)36);
	Halide::Buffer<double> buf10(b_x, 400);

	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_gemver_MEDIUM(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer(), buf06.raw_buffer(), buf07.raw_buffer(), buf09.raw_buffer(), buf08.raw_buffer(), buf10.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
