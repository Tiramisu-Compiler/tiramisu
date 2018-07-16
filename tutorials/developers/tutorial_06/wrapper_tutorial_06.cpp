#include "wrapper_tutorial_06.h"

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
    Halide::Buffer<uint8_t> reference_buf0(SIZE1, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)21);

    Halide::Buffer<uint8_t> output_buf0(SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    tiramisu::str_dump("\nBefore calling reduction.\n");
    print_buffer(output_buf0);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer());

    tiramisu::str_dump("After calling reduction.\n");
    print_buffer(output_buf0);

    compare_buffers("tutorial_06 (update)", output_buf0, reference_buf0);

    return 0;
}
