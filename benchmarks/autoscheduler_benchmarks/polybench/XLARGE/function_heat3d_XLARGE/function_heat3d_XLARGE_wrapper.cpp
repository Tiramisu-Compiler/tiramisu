#include "Halide.h"
#include "function_heat3d_XLARGE_wrapper.h"
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

	double *c_A = (double*)malloc(200*200*200*sizeof(double));
	parallel_init_buffer(c_A,200*200*200, (double)19);
	Halide::Buffer<double> A(c_A, 200,200,200);
	
	double *c_B = (double*)malloc(200*200*200*sizeof(double));
	parallel_init_buffer(c_B,200*200*200, (double)5);
	Halide::Buffer<double> B(c_B, 200,200,200);

	halide_set_custom_do_par_for(&omp_do_par_for);
	int nb_exec = get_nb_exec();

	for (int i = 0; i < nb_exec; i++) 
	{  
		auto begin = std::chrono::high_resolution_clock::now(); 
		function_heat3d_XLARGE(A.raw_buffer(), B.raw_buffer());
		auto end = std::chrono::high_resolution_clock::now(); 

		std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
	}
		std::cout << std::endl;

	return 0; 
}