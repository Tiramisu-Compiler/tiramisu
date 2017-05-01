#include "Halide.h"
#include "wrapper_tutorial_05.h"
#include "halide_image_io.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10

int main(int, char **)
{
    // Outputs
    Halide::Buffer<uint8_t> b0(NN);
    init_buffer(b0, (uint8_t)99);
    Halide::Buffer<uint8_t> b1(NN);
    init_buffer(b1, (uint8_t)99);
    Halide::Buffer<uint8_t> b2(NN, NN);
    init_buffer(b2, (uint8_t)99);
    Halide::Buffer<uint8_t> b3(NN);
    init_buffer(b3, (uint8_t)99);

    sequence(b0.raw_buffer(), b1.raw_buffer(), b2.raw_buffer(), b3.raw_buffer());

    Halide::Buffer<uint8_t> expected_b0(NN);
    init_buffer(expected_b0, (uint8_t)4);
    Halide::Buffer<uint8_t> expected_b1(NN);
    init_buffer(expected_b1, (uint8_t)3);
    Halide::Buffer<uint8_t> expected_b2(NN, NN);
    init_buffer(expected_b2, (uint8_t)2);
    Halide::Buffer<uint8_t> expected_b3(NN);
    init_buffer(expected_b3, (uint8_t)1);

    compare_buffers("tutorial_05_b0", b0, expected_b0);
    compare_buffers("tutorial_05_b1", b1, expected_b1);
    compare_buffers("tutorial_05_b2", b2, expected_b2);
    compare_buffers("tutorial_05_b3", b3, expected_b3);

    return 0;
}
