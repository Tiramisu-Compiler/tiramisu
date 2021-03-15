#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_191.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    
    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)2);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf0.raw_buffer());


    return 0;
}
