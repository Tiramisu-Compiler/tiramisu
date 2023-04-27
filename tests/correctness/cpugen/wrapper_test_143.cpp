#include "Halide.h"
#include "wrapper_test_143.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<uint32_t> input(40, 30, 20, 10);
    Halide::Buffer<uint32_t> output(40, 30, 20, 10);
    Halide::Buffer<uint32_t> reference(40, 30, 20, 10);
    for (int l = 0; l < 40; l++) {
        for (int k = 0; k < 30; k++) {
            for (int j = 0; j < 20; j++) {
                for (int i = 0; i < 10; i++) {
                    // Some seemingly random operations
                    input(l, k, j, i) = i * 43 + j * 12 + k * 4 + l;
                    output(l, k, j, i) = i * 29 + j * 31 + k * 11 + l * 3;
                    reference(l, k, j, i) = input(l, k, j, i) * 6;
                }
            }
        }
    }

    test_143(input.raw_buffer(), output.raw_buffer());
    compare_buffers("test143", output, reference);

    return 0;
}
