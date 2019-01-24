#include <tiramisu/tiramisu.h>

#include <cstdlib>
#include <iostream>

#include "mttkrp_wrapper.h"

#define PRINT_RESULTS 0

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#include "mttkrp_ref.cpp"

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<double> buf_A(SIZE, SIZE, "buf_A");
    Halide::Buffer<double> buf_A_ref(SIZE, SIZE, "buf_A_ref");
    Halide::Buffer<double> buf_B(SIZE, SIZE, SIZE, "buf_B");
    Halide::Buffer<double> buf_B_ref(SIZE, SIZE, SIZE, "buf_B_ref");
    Halide::Buffer<double> buf_C(SIZE, SIZE, "buf_C");
    Halide::Buffer<double> buf_C_ref(SIZE, SIZE, "buf_C_ref");
    Halide::Buffer<double> buf_D(SIZE, SIZE, "buf_D");
    Halide::Buffer<double> buf_D_ref(SIZE, SIZE, "buf_D_ref");

    init_buffers((double (*)[SIZE]) buf_A.raw_buffer()->host,
		 (double (*)[SIZE][SIZE]) buf_B.raw_buffer()->host,
		 (double (*)[SIZE]) buf_C.raw_buffer()->host,
		 (double (*)[SIZE]) buf_D.raw_buffer()->host);

    init_buffers((double (*)[SIZE]) buf_A_ref.raw_buffer()->host,
		 (double (*)[SIZE][SIZE]) buf_B_ref.raw_buffer()->host,
		 (double (*)[SIZE]) buf_C_ref.raw_buffer()->host,
		 (double (*)[SIZE]) buf_D_ref.raw_buffer()->host);

    for (int i = 0; i < 5; i++)
    {
	    auto start2 = std::chrono::high_resolution_clock::now();

	    ref((double (*)[SIZE]) buf_A_ref.raw_buffer()->host,
		(double (*)[SIZE][SIZE]) buf_B_ref.raw_buffer()->host,
	        (double (*)[SIZE]) buf_C_ref.raw_buffer()->host,
	        (double (*)[SIZE]) buf_D_ref.raw_buffer()->host);

	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    for (int i = 0; i < 5; i++)
    {
	    auto start1 = std::chrono::high_resolution_clock::now();

	    tiramisu_generated_code(buf_A.raw_buffer(),
				    buf_B.raw_buffer(),
				    buf_C.raw_buffer(),
				    buf_D.raw_buffer());

	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }

    if (PRINT_RESULTS)
    {
        std::cout << std::endl << "Reference buffer : ";
	print_buffer(buf_A_ref);

        std::cout << "Tiramisu buffer : ";
        print_buffer(buf_A);
    }

    compare_buffers("benchmark_" + std::string(TEST_NAME_STR), buf_A, buf_A_ref);

    print_time("performance_CPU.csv", "mttkrp",
               {"Ref", "Tiramisu"},
               {median(duration_vector_2), median(duration_vector_1)});

    return 0;
}
