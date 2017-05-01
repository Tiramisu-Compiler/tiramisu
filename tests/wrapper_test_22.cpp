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

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(SIZE0, SIZE1);
    init_buffer(reference_buf, (uint8_t)9);

    Halide::Buffer<uint8_t> output_buf0(SIZE0, SIZE1);
    Halide::Buffer<uint8_t> output_buf1(SIZE0, SIZE1);
    Halide::Buffer<uint8_t> output_buf2(SIZE0, SIZE1);
    Halide::Buffer<uint8_t> output_buf3(SIZE0, SIZE1);
    Halide::Buffer<uint8_t> output_buf4(SIZE0, SIZE1);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer(),
                            output_buf1.raw_buffer(),
                            output_buf2.raw_buffer(),
                            output_buf3.raw_buffer(),
                            output_buf4.raw_buffer());

    compare_buffers("test_" + std::string(TEST_NAME_STR), output_buf0, reference_buf);

    print_buffer(output_buf4);

    return 0;
}
