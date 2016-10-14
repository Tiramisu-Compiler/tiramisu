#include "wrapper_gemver.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <coli/utils.h>
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

    matmul(&A, &B, &C1);

    // Reference matrix multiplication
    for (int i=0; i<NN; i++)
        for (int j=0; j<NN; j++)
            for (int k=0; k<NN; k++)
                C2.host[i*NN+j] += A.host[i*NN+k]*B.host[k*NN+j];

    compare_2_2D_arrays("matmul",
                           C1.host, C2.host, NN, NN);

   return 0;
}
