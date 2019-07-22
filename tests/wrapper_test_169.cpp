#include "Halide.h"
#include "wrapper_test_169.h"

#include <tiramisu/utils.h>

void test_gemm(const std::string &name)
{
    Halide::Buffer<float> A(128, 128);
    Halide::Buffer<float> B(128, 128);
    Halide::Buffer<float> C(128, 128);
    Halide::Buffer<float> C_ref(128, 128);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
            A(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
            B(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
            C(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            C_ref(i, j) = C(i, j);
            for (int k = 0; k < 128; k++) {
                C_ref(i, j) += A(k, j) * B(i, k);
            }
        }
    }

    test_169(A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
    compare_buffers(name, C, C_ref);
}

int main(int, char **)
{
    test_gemm("test_169");

    return 0;
}
