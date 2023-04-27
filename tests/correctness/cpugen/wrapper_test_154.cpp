#include "Halide.h"
#include "wrapper_test_154.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<uint32_t> input(40, 30, 20, 10);
    Halide::Buffer<uint32_t> output(10, 20);
    Halide::Buffer<uint32_t> reference(10, 20);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
            for (int k = 0; k < 30; k++) {
                for (int l = 0; l < 40; l++) {
                    // Some psuedo-random values
                    input(l, k, j, i) = i * 43 + j * 12 + k * 4 + l;
                }
            }
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
            output(i, j) = i * 29 + j * 31;
            reference(i, j) = input(2, i, j, 3) * 30;
        }
    }

    test_154(input.raw_buffer(), output.raw_buffer());
    compare_buffers("test154", output, reference);

    return 0;
}
