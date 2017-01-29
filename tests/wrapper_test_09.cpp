#include "Halide.h"
#include "wrapper_test_09.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 20
#define MM 40

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&reference_buf, NN, MM, 167);

    buffer_t output_buf = allocate_2D_buffer(NN, MM);

    init_2D_buffer_val(&output_buf, NN, MM, 0);
    test_reduction_operator(&output_buf);
    compare_2_2D_arrays("test_reduction_operator",
                        output_buf.host, reference_buf.host, NN, MM);

   return 0;
}
