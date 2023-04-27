#include "Halide.h"
#include "wrapper_test_17.h"

#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 4
#define MM 4

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)1);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)5);

    test_tag_gpu_level(output_buf.raw_buffer());

    output_buf.copy_to_host();

    compare_buffers("test_tag_gpu_level", output_buf, reference_buf);

    return 0;
}
