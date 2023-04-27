#include "Halide.h"
#include "wrapper_test_162.h"

#include <tiramisu/utils.h>

void test_gemm(const std::string &name, int M, int N, int K, float alpha, float beta)
{
    Halide::Buffer<int32_t> sizes(3);
    Halide::Buffer<float> params(3);
    Halide::Buffer<float> A(K, M);
    Halide::Buffer<float> B(N, K);
    Halide::Buffer<float> C(N, M);
    Halide::Buffer<float> C_ref(N, M);
    sizes(0) = M;
    sizes(1) = N;
    sizes(2) = K;
    params(0) = alpha;
    params(1) = beta;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            A(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            B(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            C(i, j) = std::rand() % 10 - 5;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C_ref(i, j) = beta * C(i, j);
            for (int k = 0; k < K; k++) {
                C_ref(i, j) += alpha * A(k, j) * B(i, k);
            }
        }
    }

    test_162(sizes.raw_buffer(), params.raw_buffer(),
             A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
    compare_buffers(name, C, C_ref);
}

int main(int, char **)
{
    test_gemm("test_162_1", 100, 50, 32, 3, 4);
    test_gemm("test_162_2", 1, 2, 3, 4, 7);
    test_gemm("test_162_3", 128, 128, 128, 3, -1);
    test_gemm("test_162_4", 100, 100, 100, 1, 0);
    test_gemm("test_162_5", 100, 100, 100, 0, 1);

    return 0;
}
