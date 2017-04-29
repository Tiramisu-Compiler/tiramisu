#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_21.h"

#ifdef __cplusplus
extern "C" {
#endif

uint8_t my_external(halide_buffer_t *buf0)
{
    return 0;   //buf0->host[0];
}

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&reference_buf, SIZE0, SIZE1, 1);

    buffer_t output_buf1 = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&output_buf1, SIZE0, SIZE1, 1);

    buffer_t output_buf2 = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&output_buf2, SIZE0, SIZE1, 99);

    Halide::Buffer<uint8_t> halide_output_buf1(output_buf1);
    Halide::Buffer<uint8_t> halide_output_buf2(output_buf2);

    // Call the Tiramisu generated code
    tiramisu_generated_code(halide_output_buf1.raw_buffer(), halide_output_buf2.raw_buffer());

    compare_2_2D_arrays("test_"+std::string(TEST_NAME_STR), halide_output_buf1.data(), reference_buf.host, SIZE0, SIZE1);

    return 0;
}
