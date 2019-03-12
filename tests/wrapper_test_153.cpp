#include "Halide.h"
#include "wrapper_test_153.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<uint32_t> input(40, 30, 20, 10);
    Halide::Buffer<uint32_t> output(30, 20, 10);
    Halide::Buffer<uint32_t> reference(30, 20, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
            for (int k = 0; k < 30; k++) {
                for (int l = 0; l < 40; l++) {
                    // Some seemingly random operations
                    input(l, k, j, i) = i * 43 + j * 12 + k * 4 + l;
                }
            }
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
            for (int k = 0; k < 30; k++) {
                output(k, j, i) = i * 29 + j * 31 + k * 13;
                reference(k, j, i) = input(k, 1, j, i) * 18;
            }
        }
    }

    test_153(input.raw_buffer(), output.raw_buffer());
    compare_buffers("test153", output, reference);

    return 0;
}
