#include "Halide.h"
#include "wrapper_test_168.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    Halide::Buffer<float> A(256, 128);
    Halide::Buffer<float> B(128, 256);
    Halide::Buffer<float> B_ref(128, 256);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 256; j++) {
            A(j, i) = std::rand() % 10 - 5;
            B(i, j) = std::rand() % 10 - 5;
            B_ref(i, j) = A(j, i);
        }

    test_168(A.raw_buffer(), B.raw_buffer());
    compare_buffers("test_168", B, B_ref);

    return 0;
}
