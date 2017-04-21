#include "Halide.h"
#include "wrapper_tutorial_05.h"
#include "halide_image_io.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10

int main(int, char**)
{
    buffer_t b0 = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b0, NN, 1);

    buffer_t b1 = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b1, NN, 1);

    buffer_t b2 = allocate_2D_buffer(NN, NN);
    init_2D_buffer_val(&b2, NN, NN, 1);

    buffer_t b3 = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b3, NN, 1);

    Halide::Buffer<uint8_t> b0_buf(b0);
    Halide::Buffer<uint8_t> b1_buf(b1);
    Halide::Buffer<uint8_t> b2_buf(b2);
    Halide::Buffer<uint8_t> b3_buf(b3);

    sequence(b0_buf.raw_buffer(), b1_buf.raw_buffer(), b2_buf.raw_buffer(), b3_buf.raw_buffer());

   return 0;
}
