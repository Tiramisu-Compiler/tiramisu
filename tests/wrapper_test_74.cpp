#include "Halide.h"
#include "wrapper_test_74.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char **)
{
    Halide::Buffer<int32_t> reference_buf(NN, MM);

    for (int i = 0; i < NN; i++)
        for (int j = 0; j < MM; j++)
            // Stride of the first dim is 1, so flip j and i
            reference_buf(j, i) = j + i / 2;

    Halide::Buffer<int32_t> output_buf(NN, MM);

    test_inlining_2(output_buf.raw_buffer());
    compare_buffers("test_inlining_2", output_buf, reference_buf);

    return 0;
}
