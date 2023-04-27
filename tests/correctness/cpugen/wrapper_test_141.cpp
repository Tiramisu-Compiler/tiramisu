#include "Halide.h"
#include "wrapper_test_141.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<uint8_t> input(200, 100);
    init_buffer(input, (uint8_t)13);

    Halide::Buffer<uint8_t> output(200, 100);
    init_buffer(output, (uint8_t)7);

    Halide::Buffer<uint8_t> reference(200, 100);
    init_buffer(reference, (uint8_t)26);

    test_141(input.raw_buffer(), output.raw_buffer());
    compare_buffers("init", output, reference);

    return 0;
}
