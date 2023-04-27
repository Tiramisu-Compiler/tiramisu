#include "Halide.h"
#include "wrapper_test_172.h"

#include <tiramisu/utils.h>

#define NN 10

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN);
    init_buffer(reference_buf, (uint8_t)7);

    Halide::Buffer<uint8_t> output_buf(NN);
    init_buffer(output_buf, (uint8_t)13);

    func(output_buf.raw_buffer());
    compare_buffers("computation_constructor", output_buf, reference_buf);

    return 0;
}
