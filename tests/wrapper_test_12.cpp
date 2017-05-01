#include "Halide.h"
#include "wrapper_test_12.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char**)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)7);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)13);

    assign_7_to_10x10_2D_array_with_auto_unrolling(output_buf.raw_buffer());

    compare_buffers("assign_7_to_10x10_2D_array_with_auto_unrolling", output_buf, reference_buf);

   return 0;
}
