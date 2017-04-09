#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_20.h"

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&reference_buf, SIZE0, SIZE1, 4);

    buffer_t output_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&output_buf, SIZE0, SIZE1, 99);

    // Call the Tiramisu generated code
    TEST_NAME(&output_buf);
    compare_2_2D_arrays("test_"+std::string(TEST_NAME_STR), output_buf.host, reference_buf.host, SIZE0, SIZE1);
    print_2D_buffer(&reference_buf, SIZE0, SIZE1);
    print_2D_buffer(&output_buf, SIZE0, SIZE1);

   return 0;
}
