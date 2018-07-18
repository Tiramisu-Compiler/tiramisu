#include "Halide.h"
#include "wrapper_tutorial_04gpu.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000

int main(int, char **)
{
    Halide::Buffer<uint8_t> A_buf(NN, NN);
    Halide::Buffer<uint8_t> B_buf(NN, NN);
    // Initialize matrices with pseudorandom values:
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            A_buf(j, i) = (i + 3) * (j + 1);
            B_buf(j, i) = (i + 1) * j + 2;
        }
    }

    // Output
    Halide::Buffer<uint8_t> C1_buf(NN, NN);

    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());

    // Reference matrix multiplication
    Halide::Buffer<uint8_t> C2_buf(NN, NN);
    init_buffer(C2_buf, (uint8_t)0);

    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            for (int k = 0; k < NN; k++) {
                // Note that indices are flipped (see tutorial 2)
                C2_buf(j, i) += A_buf(k, i) * B_buf(j, k);
            }
        }
    }

    compare_buffers("matmul", C1_buf, C2_buf);

    return 0;
}
