#include "wrapper_spmv.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <coli/utils.h>
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

    spmv(&b_row_start, &b_col_idx, &b_values, &b_x, &b_y);

    //TODO compute reference spmv and compare with the Coli spmv

   return 0;
}
