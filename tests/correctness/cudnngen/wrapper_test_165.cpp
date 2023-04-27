#include "Halide.h"
#include "wrapper_test_165.h"

#include <tiramisu/utils.h>

void test_gemm(const std::string &name, int M, int N, int K)
{
    Halide::Buffer<int32_t> sizes(3);
    Halide::Buffer<float> A(K, M);
    Halide::Buffer<float> B(K, N);
    Halide::Buffer<float> C(N, M);
    Halide::Buffer<float> C_ref(N, M);
    sizes(0) = M;
    sizes(1) = N;
    sizes(2) = K;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            B(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_ref(j, i) = 0;
            for (int k = 0; k < K; k++) {
                C_ref(j, i) += A(k, i) * B(k, j);
            }
        }
    }

    test_165(sizes.raw_buffer(), A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
    compare_buffers(name, C, C_ref);
}

int main(int, char **)
{
    test_gemm("test_165_1", 100, 50, 32);
    test_gemm("test_165_2", 1, 1, 1);
    test_gemm("test_165_3", 162, 1, 113);
    test_gemm("test_165_4", 16, 100, 13);
    test_gemm("test_165_5", 2, 2, 1000);
    return 0;
}
