#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_94.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;

    Halide::Buffer<float> reference_buf0(SIZE, SIZE, 3, "reference_buf0");
    init_buffer(reference_buf0, (float)2);

    Halide::Buffer<float> output_buf0(SIZE, SIZE, 3, "output_buf0");
    init_buffer(output_buf0, (float)0);

    for (int i=0; i<100; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
	tiramisu_generated_code(output_buf0.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    print_time("performance_CPU.csv", "PLDI",
               {"Tiramisu", "Tiramisu"},
               {median(duration_vector_1), median(duration_vector_1)});

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}
