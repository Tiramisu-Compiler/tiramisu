#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_53.h"

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

    Halide::Buffer<uint8_t> input_buf0(SIZE1, SIZE1, "input_buf0");
    init_buffer(input_buf0, (uint8_t)5);

    Halide::Buffer<uint8_t> input_buf1(SIZE1, SIZE1, "input_buf1");
    init_buffer(input_buf1, (uint8_t)5);

    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    Halide::Buffer<uint8_t> output_buf1(SIZE1, SIZE1, "output_buf1");
    init_buffer(output_buf1, (uint8_t)0);

    tiramisu_generated_code(input_buf0.raw_buffer(),
                            input_buf1.raw_buffer(),
                            output_buf0.raw_buffer(),
                            output_buf1.raw_buffer());

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}
