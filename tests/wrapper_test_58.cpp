#include "Halide.h"
#include "wrapper_test_58.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 100

int main(int, char **)
{
    Halide::Buffer<int32_t> tiramisu_output(NN, NN);
    Halide::Buffer<int32_t> reference_output(NN, NN);

    init_buffer(reference_output, (int32_t) 6);

    scheduled_with_before(tiramisu_output.raw_buffer());

    compare_buffers("test_scheduled_with_before", tiramisu_output, reference_output);

    return 0;
}
