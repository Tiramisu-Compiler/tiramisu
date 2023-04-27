#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_71.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

// We assume that the increment is 1.
void reference_saxpy(int N1, float alpha, float *A, float *B)
{
	for (int i=0; i<N1; i++)
		B[i] = alpha*A[i] + B[i];
}

int main(int, char **)
{
    Halide::Buffer<float> a(1, "a");
    Halide::Buffer<float> x(SIZE, "x");
    Halide::Buffer<float> y_ref(SIZE, "y_ref");
    Halide::Buffer<float> y(SIZE, "y");

    init_buffer(x, (float)1);
    init_buffer(y, (float)1);
    init_buffer(y_ref, (float)1);
    init_buffer(a, (float)1);

    reference_saxpy(SIZE, 1, x.data(), y_ref.data());
    tiramisu_generated_code(a.raw_buffer(), x.raw_buffer(), y.raw_buffer());

    compare_buffers("test_" + std::string(TEST_NUMBER_STR) + "_"  + std::string(TEST_NAME_STR), y, y_ref);

    return 0;
}
