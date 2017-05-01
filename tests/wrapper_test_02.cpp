#include "Halide.h"
#include "wrapper_test_02.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 100
#define MM 100

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)7);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)13);

    assign_7_to_100x100_2D_array_with_tiling_parallelism(output_buf.raw_buffer());

    compare_buffers("assign_7_to_100x100_2D_array_with_tiling_parallelism", output_buf, reference_buf);

    return 0;
}
