#include "Halide.h"
#include "function_3mm_MINI_wrapper.h"
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

	double *b_A = (double*)malloc(16*20*sizeof(double));
	parallel_init_buffer(b_A, 16*20, (double)19);
	Halide::Buffer<double> buf01(b_A, 16, 20);

	double *b_B = (double*)malloc(20*18* sizeof(double));
	parallel_init_buffer(b_B, 20*18, (double)36);
	Halide::Buffer<double> buf02(b_B, 20,18);

	double *b_C = (double*)malloc(18*24*sizeof(double));
	parallel_init_buffer(b_C, 18*24, (double)19);
	Halide::Buffer<double> buf03(b_C, 18, 24);

	double *b_D = (double*)malloc(24*22* sizeof(double));
	parallel_init_buffer(b_D, 24*22, (double)36);
	Halide::Buffer<double> buf04(b_D, 24, 22);

	double *b_E = (double*)malloc(16*22*sizeof(double));
	parallel_init_buffer(b_E, 16*22, (double)19);
	Halide::Buffer<double> buf05(b_E, 16, 22);

	
	



	halide_set_custom_do_par_for(&omp_do_par_for);
	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_3mm_MINI(buf01.raw_buffer(), buf02.raw_buffer(), buf03.raw_buffer(), buf04.raw_buffer(), buf05.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
