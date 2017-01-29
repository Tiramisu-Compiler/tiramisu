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

    sequence(&b0, &b1, &b2, &b3);

   return 0;
}
