#include "Halide.h"
#include "wrapper_test_07.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10

int main(int, char**)
{
    Halide::Buffer<uint8_t> reference_buf(NN);
    init_buffer(reference_buf, (uint8_t)4);

    Halide::Buffer<uint8_t> output_buf(NN);
    init_buffer(output_buf, (uint8_t)0);

    test_duplication(output_buf.raw_buffer());

    compare_buffers("test_duplication", output_buf, reference_buf);

   return 0;
}
