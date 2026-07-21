#include "Halide.h"
#include "function_gemver_XLARGE_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>
#include <omp.h>


int omp_do_par_for(void *user_context, int (*f)(void *, int, uint8_t *), int min, int extent, uint8_t *state) {
    int exit_status = 0;
    #pragma omp parallel for    
    for (int idx=min; idx<min+extent; idx++){
      int job_status = halide_do_task(user_context, f, idx, state);
      if (job_status) exit_status = job_status;
    }
    return exit_status;
}

using namespace std::chrono;
using namespace std;      

int main(int, char **argv)
{

	double *b_A = (double*)malloc(4000*4000*sizeof(double));
	parallel_init_buffer(b_A,4000*4000, (double)19);
	Halide::Buffer<double> buf01(b_A, 4000, 4000);

	double *b_u1 = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_u1, 4000, (double)36);
	Halide::Buffer<double> buf02(b_u1, 4000);

	double *b_u2 = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_u2, 4000, (double)36);
	Halide::Buffer<double> buf03(b_u2, 4000);

	double *b_v1 = (double*)malloc(4000*sizeof(double));
	parallel_init_buffer(b_v1,4000, (double)19);
	Halide::Buffer<double> buf04(b_v1, 4000);

	double *b_v2 = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_v2, 4000, (double)36);
	Halide::Buffer<double> buf05(b_v2, 4000);

	double *b_y = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_y, 4000, (double)36);
	Halide::Buffer<double> buf06(b_y, 4000);

	double *b_z = (double*)malloc(4000*sizeof(double));
	parallel_init_buffer(b_z,4000, (double)19);
	Halide::Buffer<double> buf07(b_z, 4000);

	double *b_w = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_w, 4000, (double)36);
	Halide::Buffer<double> buf08(b_w, 4000);

	double *b_A_hat = (double*)malloc(4000*4000*sizeof(double));
	parallel_init_buffer(b_A_hat,4000*4000, (double)19);
	Halide::Buffer<double> buf09(b_A_hat, 4000, 4000);

	double *b_x = (double*)malloc(4000* sizeof(double));
	parallel_init_buffer(b_x, 4000, (double)36);
	Halide::Buffer<double> buf10(b_x, 4000);

	halide_set_custom_do_par_for(&omp_do_par_for);
	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_gemver_XLARGE(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer(), buf06.raw_buffer(), buf07.raw_buffer(), buf09.raw_buffer(), buf08.raw_buffer(), buf10.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
