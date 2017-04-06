#include "Halide.h"
#include "wrapper_test_17.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 4
#define MM 4

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&reference_buf, NN, MM, 1);

    buffer_t output_buf = allocate_2D_buffer(NN, MM);

    init_2D_buffer_val(&output_buf, NN, MM, 5);
    test_tag_gpu_level(&output_buf);
    compare_2_2D_arrays("test_tag_gpu_level",
                        output_buf.host, reference_buf.host, NN, MM);

   return 0;
}
