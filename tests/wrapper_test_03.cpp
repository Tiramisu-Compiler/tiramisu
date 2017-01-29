#include "Halide.h"
#include "wrapper_test_03.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000
#define MM 1000

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&reference_buf, NN, MM, 7);

    buffer_t output_buf = allocate_2D_buffer(NN, MM);

    init_2D_buffer_val(&output_buf, NN, MM, 13);
    assign_7_to_1000x1000_2D_array_with_tiling_parallelism(&output_buf);
    compare_2_2D_arrays("assign_7_to_1000x1000_2D_array_with_tiling_parallelism",
                        output_buf.host, reference_buf.host, NN, MM);

    return 0;
}
