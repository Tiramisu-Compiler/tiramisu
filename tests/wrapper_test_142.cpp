#include "Halide.h"
#include "wrapper_test_142.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<uint32_t> input(200, 100);
    Halide::Buffer<uint32_t> output(200, 100);
    Halide::Buffer<uint32_t> reference(200, 100);
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 100; j++) {
            // Some seemingly random operations
            input(i, j) = i * 43 + j * 12 + 3;
            output(i, j) = i * 23 + j * 15 + 1;
            reference(i, j) = input(i, j) * 22 + input(0, j) * 22 + 71;
        }
    }

    test_142(input.raw_buffer(), output.raw_buffer());
    compare_buffers("test142", output, reference);

    return 0;
}
