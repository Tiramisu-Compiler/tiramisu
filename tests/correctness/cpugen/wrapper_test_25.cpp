#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_25.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf1(SIZE0);
    init_buffer(reference_buf1, (uint8_t)1);

    Halide::Buffer<uint8_t> output_buf0(SIZE0);
    init_buffer(output_buf0, (uint8_t)0);
    output_buf0.raw_buffer()->host[SIZE0-2] = (uint8_t)1;
    output_buf0.raw_buffer()->host[SIZE0-1] = (uint8_t)1;

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer());

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf1);

    return 0;
}
