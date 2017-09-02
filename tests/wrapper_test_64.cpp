#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_64.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE1, SIZE1, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)5);

    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    tiramisu_generated_code(output_buf0.raw_buffer());

    compare_buffers("test_" + std::string(TEST_NUMBER_STR) + "_"  + std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}
