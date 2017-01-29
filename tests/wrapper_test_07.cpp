#include "Halide.h"
#include "wrapper_test_07.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10

int main(int, char**)
{
    buffer_t reference_buf = allocate_1D_buffer(NN);
    init_1D_buffer_val(&reference_buf, NN, 4);

    buffer_t output_buf = allocate_1D_buffer(NN);

    init_1D_buffer_val(&output_buf, NN, 0);
    test_duplication(&output_buf);
    compare_2_1D_arrays("test_duplication",
                        output_buf.host, reference_buf.host, NN);

   return 0;
}
