#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "baryon_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#include "baryon_ref.cpp"

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> buf_res2(1, "buf_res2");
    Halide::Buffer<float> buf_res2_ref(1, "buf_res2_ref");
    Halide::Buffer<float> buf_S(BARYON_P, BARYON_P, BARYON_P, BARYON_N, BARYON_N, BARYON_N, BARYON_P, "buf_S");
    Halide::Buffer<float> buf_wp(BARYON_N, BARYON_P, BARYON_P, BARYON_P, BARYON_P, BARYON_P, BARYON_P, "buf_wp");

    init_buffer(buf_S, (float)1);
    init_buffer(buf_wp, (float)1);

    for (int i = 0; i < NB_TESTS; i++)
    {
    	    init_buffer(buf_res2_ref, (float)0);
	    auto start2 = std::chrono::high_resolution_clock::now();
	    ref((float *) buf_res2_ref.raw_buffer()->host, (float (*)[BARYON_P][BARYON_P][BARYON_N][BARYON_N][BARYON_N][BARYON_P]) buf_S.raw_buffer()->host,
			    				   (float (*)[BARYON_P][BARYON_P][BARYON_P][BARYON_P][BARYON_P][BARYON_P]) buf_wp.raw_buffer()->host);
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(buf_res2, (float)0);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    tiramisu_generated_code(buf_res2.raw_buffer(), buf_S.raw_buffer(), buf_wp.raw_buffer());
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }

    compare_buffers("benchmark_" + std::string(TEST_NAME_STR), buf_res2, buf_res2_ref);

    print_time("performance_CPU.csv", "baryon",
               {"Ref", "Tiramisu"},
               {median(duration_vector_2), median(duration_vector_1)});

    return 0;
}
