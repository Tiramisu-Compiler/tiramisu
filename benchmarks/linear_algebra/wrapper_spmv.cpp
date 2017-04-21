#include "wrapper_spmv.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000

int main(int, char**)
{
    buffer_t b_row_start = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b_row_start, NN, 1);

    buffer_t b_col_idx = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b_col_idx, NN, 1);

    buffer_t b_values = allocate_1D_buffer(NN);
    init_1D_buffer_val(&b_values, NN, 1);

    buffer_t b_x = allocate_1D_buffer(NN*NN);
    init_1D_buffer_val(&b_x, NN*NN, 1);

    buffer_t b_y = allocate_1D_buffer(NN*NN);
    init_1D_buffer_val(&b_y, NN*NN, 1);

    Halide::Buffer<uint8_t> b_row_start_buf(b_row_start);
    Halide::Buffer<uint8_t> b_col_idx_buf(b_col_idx);
    Halide::Buffer<uint8_t> b_values_buf(b_values);
    Halide::Buffer<uint8_t> b_x_buf(b_x);
    Halide::Buffer<uint8_t> b_y_buf(b_y);

    spmv(b_row_start_buf.raw_buffer(), b_col_idx_buf.raw_buffer(), b_values_buf.raw_buffer(), b_x_buf.raw_buffer(), b_y_buf.raw_buffer());

    //TODO: compute reference spmv and compare with the Tiramisu spmv

   return 0;
}
