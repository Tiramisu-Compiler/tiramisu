#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_96.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    // TODO: create Halide buffers for reference and output
    Halide::Buffer<int64_t> reference_buffer(SIZE0), output_buffer(SIZE0);

    for (int64_t i = 0; i < SIZE0; i ++)
    {
        reference_buffer(i) = i*i - SIZE0;
    }

    tiramisu_generated_code(output_buffer.raw_buffer());

    // TODO: do assertions. use TEST_ID_STR to name the test.
    compare_buffers(TEST_ID_STR, output_buffer, reference_buffer);

    return 0;
}
