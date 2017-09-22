#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_84.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<int32_t> input(SIZE);
    Halide::Buffer<int32_t> output(SIZE);
    Halide::Buffer<int32_t> reference(SIZE);
    init_buffer(input, (int32_t)0);
    init_buffer(reference, (int32_t)11);

    tiramisu_generated_code(input.raw_buffer(), output.raw_buffer());

    compare_buffers(TEST_ID_STR, output, reference);

    return 0;
}
