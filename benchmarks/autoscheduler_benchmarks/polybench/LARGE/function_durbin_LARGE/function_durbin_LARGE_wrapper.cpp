#include "Halide.h"
#include "function_durbin_LARGE_wrapper.h"
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

	double *b_A = (double*)malloc(2000*2000*sizeof(double));
	parallel_init_buffer(b_A, 2000*2000, (double)19);
	Halide::Buffer<double> buf01(b_A, 2000, 2000);

	double *b_C = (double*)malloc(2000* sizeof(double));
	parallel_init_buffer(b_C, 2000, (double)36);
	Halide::Buffer<double> buf02(b_C, 2000);

	double *b_D = (double*)malloc(2000*sizeof(double));
	parallel_init_buffer(b_D, 2000, (double)19);
	Halide::Buffer<double> buf03(b_D, 2000);

	double *b_E = (double*)malloc(2000* sizeof(double));
	parallel_init_buffer(b_E, 2000, (double)36);
	Halide::Buffer<double> buf04(b_E, 2000);

	double *b_F = (double*)malloc(2000*sizeof(double));
	parallel_init_buffer(b_F, 2000, (double)19);
	Halide::Buffer<double> buf05(b_F, 2000);




	halide_set_custom_do_par_for(&omp_do_par_for);
	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_durbin_LARGE(buf03.raw_buffer(), buf02.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}
