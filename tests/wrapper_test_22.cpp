#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_22.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&reference_buf, SIZE0, SIZE1, 9);

    buffer_t output_buf0 = allocate_2D_buffer(SIZE0, SIZE1);
    buffer_t output_buf1 = allocate_2D_buffer(SIZE0, SIZE1);
    buffer_t output_buf2 = allocate_2D_buffer(SIZE0, SIZE1);
    buffer_t output_buf3 = allocate_2D_buffer(SIZE0, SIZE1);
    buffer_t output_buf4 = allocate_2D_buffer(SIZE0, SIZE1);
    init_2D_buffer_val(&output_buf0, SIZE0, SIZE1, 0);
    init_2D_buffer_val(&output_buf1, SIZE0, SIZE1, 0);
    init_2D_buffer_val(&output_buf2, SIZE0, SIZE1, 0);
    init_2D_buffer_val(&output_buf3, SIZE0, SIZE1, 0);
    init_2D_buffer_val(&output_buf4, SIZE0, SIZE1, 0);
    Halide::Buffer<uint8_t> halide_output_buf0(output_buf0);
    Halide::Buffer<uint8_t> halide_output_buf1(output_buf1);
    Halide::Buffer<uint8_t> halide_output_buf2(output_buf2);
    Halide::Buffer<uint8_t> halide_output_buf3(output_buf3);
    Halide::Buffer<uint8_t> halide_output_buf4(output_buf4);


    // Call the Tiramisu generated code
    tiramisu_generated_code(halide_output_buf0.raw_buffer(),
                            halide_output_buf1.raw_buffer(),
                            halide_output_buf2.raw_buffer(),
                            halide_output_buf3.raw_buffer(),
                            halide_output_buf4.raw_buffer());

    compare_2_2D_arrays("test_"+std::string(TEST_NAME_STR), halide_output_buf4.data(), reference_buf.host, SIZE0, SIZE1);

    print_2D_array(halide_output_buf4.data(), SIZE0, SIZE1);

    return 0;
}
