#include "Halide.h"
#include "wrapper_test_18.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(SIZE0, SIZE1);
    init_buffer(reference_buf, (uint8_t)1);

    Halide::Buffer<uint8_t> output_buf(SIZE0, SIZE1);
    init_buffer(output_buf, (uint8_t)5);

    // Call the Tiramisu generated code
    TEST_NAME(output_buf.raw_buffer());

    compare_buffers("test_" + std::string(TEST_NAME_STR), output_buf, reference_buf);

    return 0;
}
