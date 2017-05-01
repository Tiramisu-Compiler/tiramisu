#include "Halide.h"
#include "wrapper_test_04.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000
#define MM 1000

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)20);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)99);

    test_let_stmt(output_buf.raw_buffer());

    compare_buffers("test_let_stmt", output_buf, reference_buf);

    return 0;
}
