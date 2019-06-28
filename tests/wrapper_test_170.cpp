#include "Halide.h"
#include "wrapper_test_170.h"

#include <tiramisu/utils.h>

#define N 128
#define M 128
#define K 128

void test_gemm(const std::string &name)
{
    Halide::Buffer<float> A(K, N);
    Halide::Buffer<float> B(M, K);
    Halide::Buffer<float> C(M, N);
    Halide::Buffer<float> C_ref(M, N);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            A(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            B(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_ref(i, j) = C(i, j);
            for (int k = 0; k < K; k++) {
                C_ref(i, j) += A(k, j) * B(i, k);
            }
        }
    }

    test_170(A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
    compare_buffers(name, C, C_ref);
}

int main(int, char **)
{
    test_gemm("test_170");

    return 0;
}
