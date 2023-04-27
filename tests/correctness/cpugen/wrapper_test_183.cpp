#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_183.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
 
    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");

    // Call the Tiramisu generated code
    /* Since we are testing that the legality check is false 
        the result should also be diffrent
        The test that is being done is an assert(check_legality == false) inside the function
    */
    tiramisu_generated_code(output_buf0.raw_buffer());


    return 0;
}
