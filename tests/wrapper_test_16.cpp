#include "Halide.h"
#include "wrapper_test_16.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&reference_buf, NN, MM, 1);

    buffer_t output_buf = allocate_2D_buffer(NN, MM);

    init_2D_buffer_val(&output_buf, NN, MM, 5);
    test_access_parsing(&output_buf);
    compare_2_2D_arrays("test_access_parsing",
                        output_buf.host, reference_buf.host, NN, MM);

   return 0;
}
