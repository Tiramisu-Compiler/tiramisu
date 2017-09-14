#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_75.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<int32_t> output(SIZE0, SIZE0);
    Halide::Buffer<int32_t> reference(SIZE0, SIZE0);

    for (int i = 0; i < SIZE0; i++)
        for (int j = 0; j < SIZE0; j++)
            reference(i, j) = (i + j)*(i + j);

    tiramisu_generated_code(output.raw_buffer());

    compare_buffers(TEST_ID_STR, output, reference);

    return 0;
}
