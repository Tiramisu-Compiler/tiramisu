#include "Halide.h"
#include "wrapper_test_155.h"

#include <cstdlib>

#include <tiramisu/utils.h>

int main(int, char **)
{
    int M = 10, N = 20, K = 30;
    Halide::Buffer<int32_t> sizes(3);
    sizes(0) = M;
    sizes(1) = N;
    sizes(2) = K;
    Halide::Buffer<int32_t> A(N, M);
    Halide::Buffer<int32_t> B(K, N);
    Halide::Buffer<int32_t> C(K, M);
    Halide::Buffer<int32_t> C_ref(K, M);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A(j, i) = std::rand() % 10 - 5;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B(j, i) = std::rand() % 10 - 5;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            C(j, i) = std::rand() % 10 - 5;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            C_ref(k, i) = 0;
            for (int j = 0; j < N; j++) {
                C_ref(k, i) += A(j, i) * B(k, j);
            }
        }
    }

    test_155(sizes.raw_buffer(),
             A.raw_buffer(),
             B.raw_buffer(),
             C.raw_buffer());
    compare_buffers("test155", C, C_ref);

    return 0;
}
