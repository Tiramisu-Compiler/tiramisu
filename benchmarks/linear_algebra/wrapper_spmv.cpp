#include "wrapper_spmv.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 1000

int main(int, char**)
{
    Halide::Buffer<uint8_t> b_row_start_buf(NN);
    init_buffer(b_row_start_buf, 1);
    Halide::Buffer<uint8_t> b_col_idx_buf(NN);
    init_buffer(b_col_idx_buf, 1);
    Halide::Buffer<uint8_t> b_values_buf(NN);
    init_buffer(b_values_buf, 1);
    Halide::Buffer<uint8_t> b_x_buf(NN*NN);
    init_buffer(b_x_buf, 1);
    Halide::Buffer<uint8_t> b_y_buf(NN*NN);
    init_buffer(b_y_buf, 1);

    spmv(b_row_start_buf.raw_buffer(), b_col_idx_buf.raw_buffer(), b_values_buf.raw_buffer(), b_x_buf.raw_buffer(), b_y_buf.raw_buffer());

    //TODO: compute reference spmv and compare with the Tiramisu spmv

   return 0;
}
