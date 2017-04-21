#include "Halide.h"
#include "wrapper_test_18.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&reference_buf, SIZE0, SIZE1, 1);

    buffer_t output_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&output_buf, SIZE0, SIZE1, 5);
    Halide::Buffer<uint8_t> halide_output_buf(output_buf);

    // Call the Tiramisu generated code
    TEST_NAME(halide_output_buf.raw_buffer());

    compare_2_2D_arrays("test_"+std::string(TEST_NAME_STR), halide_output_buf.data(), reference_buf.host, SIZE0, SIZE1);

   return 0;
}
