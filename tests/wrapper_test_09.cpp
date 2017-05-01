#include "Halide.h"
#include "wrapper_test_09.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 20
#define MM 40

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)167);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)0);

    test_reduction_operator(output_buf.raw_buffer());

    compare_buffers("test_reduction_operator", output_buf, reference_buf);

    return 0;
}
