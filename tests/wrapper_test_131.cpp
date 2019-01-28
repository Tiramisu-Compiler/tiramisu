#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_131.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE1, SIZE1, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)2);

    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer());

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}
