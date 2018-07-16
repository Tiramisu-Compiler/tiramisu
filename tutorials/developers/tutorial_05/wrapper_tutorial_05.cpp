#include "wrapper_tutorial_05.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE0, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)10);

    Halide::Buffer<uint8_t> output_buf0(SIZE0, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    Halide::Buffer<uint8_t> input_buf0(SIZE1, "input_buf0");
    init_buffer(input_buf0, (uint8_t)1);

    tiramisu::str_dump("Before calling reduction.\n");
    print_buffer(input_buf0);
    print_buffer(output_buf0);
    print_buffer(reference_buf0);

    // Call the Tiramisu generated code
    tiramisu_generated_code(input_buf0.raw_buffer(), output_buf0.raw_buffer());

    tiramisu::str_dump("After calling reduction.\n");
    print_buffer(input_buf0);
    print_buffer(output_buf0);
    print_buffer(reference_buf0);

    compare_buffers("tutorial_05 (reduction)", output_buf0, reference_buf0);

    return 0;
}
