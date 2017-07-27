#include "Halide.h"
#include "wrapper_test_56.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 100

int clamp(int k) {
    if (k < 0)
        return 0;
    if (k >= NN)
        return NN - 1;
    return k;
}

int main(int, char **)
{
    Halide::Buffer<uint8_t> input_buf(NN, NN);
    Halide::Buffer<uint8_t> tiramisu_output(NN, NN);
    Halide::Buffer<uint8_t> reference_output(NN, NN);

    // Initialize the input buffer with different values
    for (int i = 0; i < NN; i++)
        for (int j = 0; j < NN; j++)
            input_buf(i, j) = (uint8_t) (i + j);


    init_buffer(reference_output, (uint8_t)0);

    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            for (int p = -1; p < 2; p++) {
                for (int q = -1; q < 2; q++) {
                    reference_output(i, j) += input_buf(clamp(i + p), clamp(j + q));
                }
            }
            reference_output(i, j) /= (uint8_t) 9;
        }
    }


    blur_100x100_2D_array_with_tiling_parallelism(input_buf.raw_buffer(),
                                                  tiramisu_output.raw_buffer());

    compare_buffers("test_clamp_blur", tiramisu_output, reference_output);

    return 0;
}
