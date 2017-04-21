#include "wrapper_gemver.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000

int main(int, char**)
{
    buffer_t A = allocate_2D_buffer(NN, NN);
    init_2D_buffer_val(&A, NN, NN, 1);

    buffer_t B = allocate_2D_buffer(NN, NN);
    init_2D_buffer_val(&B, NN, NN, 1);

    buffer_t C1 = allocate_2D_buffer(NN, NN);
    init_2D_buffer_val(&C1, NN, NN, 0);

    buffer_t C2 = allocate_2D_buffer(NN, NN);
    init_2D_buffer_val(&C2, NN, NN, 0);

    Halide::Buffer<uint8_t> A_buf(A);
    Halide::Buffer<uint8_t> B_buf(B);
    Halide::Buffer<uint8_t> C1_buf(C1);
    Halide::Buffer<uint8_t> C2_buf(C2);

    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());

    // Reference matrix multiplication
    for (int i=0; i<NN; i++)
        for (int j=0; j<NN; j++)
            for (int k=0; k<NN; k++)
                C2_buf.data()[i*NN+j] += A_buf.data()[i*NN+k]*B_buf.data()[k*NN+j];

    compare_2_2D_arrays("matmul", C1_buf.data(), C2_buf.data(), NN, NN);

    return 0;
}
