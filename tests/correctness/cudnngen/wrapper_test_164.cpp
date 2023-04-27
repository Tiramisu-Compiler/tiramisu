#include "Halide.h"
#include "wrapper_test_164.h"

#include <tiramisu/utils.h>

void test_gemm(const std::string &name,
               int M, int N, int K,
               float alpha, float beta,
               int rowsA, int colsA,
               int rowsB, int colsB,
               int rowsC, int colsC,
               int offsetA, int offsetB, int offsetC,
               bool transposeA, bool transposeB)
{
    Halide::Buffer<int32_t> sizes(12);
    Halide::Buffer<float> params(2);
    Halide::Buffer<bool> transposes(2);
    Halide::Buffer<float> A(colsA, rowsA);
    Halide::Buffer<float> B(colsB, rowsB);
    Halide::Buffer<float> C(colsC, rowsC);
    Halide::Buffer<float> C_ref(colsC, rowsC);
    sizes(0) = M;
    sizes(1) = N;
    sizes(2) = K;
    sizes(3) = rowsA;
    sizes(4) = colsA;
    sizes(5) = rowsB;
    sizes(6) = colsB;
    sizes(7) = rowsC;
    sizes(8) = colsC;
    sizes(9) = offsetA;
    sizes(10) = offsetB;
    sizes(11) = offsetC;
    params(0) = alpha;
    params(1) = beta;
    transposes(0) = transposeA;
    transposes(1) = transposeB;

    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < colsA; j++)
            A(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < rowsB; i++)
        for (int j = 0; j < colsB; j++)
            B(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < rowsC; i++)
        for (int j = 0; j < colsC; j++)
            C(j, i) = C_ref(j, i) = std::rand() % 10 - 5;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_ref(j + offsetC % colsC, i + offsetC / colsC) *= beta;
            for (int k = 0; k < K; k++) {
                float a = transposeA ?
                              A(i + offsetA % colsA, k + offsetA / colsA) :
                              A(k + offsetA % colsA, i + offsetA / colsA);
                float b = transposeB ?
                              B(k + offsetB % colsB, j + offsetB / colsB) :
                              B(j + offsetB % colsB, k + offsetB / colsB);
                C_ref(j + offsetC % colsC, i + offsetC / colsC) += alpha * a * b;
            }
        }
    }

    test_164(sizes.raw_buffer(), params.raw_buffer(), transposes.raw_buffer(),
             A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
    compare_buffers(name, C, C_ref);
}

int main(int, char **)
{
    test_gemm("test_164_basic",
              100, 50, 32,
              3, 4,
              100, 32,
              32, 50,
              100, 50,
              0, 0, 0,
              false, false);
    test_gemm("test_164_ldA",
              100, 50, 32,
              3, 4,
              1000, 1000,
              32, 50,
              100, 50,
              0, 0, 0,
              false, false);
    test_gemm("test_164_ld",
              15, 107, 11,
              75, 14,
              1000, 3000,
              2000, 300,
              164, 132,
              0, 0, 0,
              false, false);
    test_gemm("test_164_trA",
              51, 17, 21,
              75, 15,
              1000, 2000,
              2000, 300,
              164, 232,
              0, 0, 0,
              true, false);
    test_gemm("test_164_trB",
              51, 17, 21,
              75, 15,
              100, 20000,
              222, 131,
              100, 123,
              0, 0, 0,
              false, true);
    test_gemm("test_164_offset",
              51, 17, 21,
              75, 15,
              100, 20000,
              222, 131,
              100, 123,
              20000 * 4 + 3, 131 * 5 + 6, 123 * 4 + 23,
              false, false);
    test_gemm("test_164_all",
              51, 17, 21,
              75, 15,
              100, 20000,
              222, 131,
              100, 123,
              20000 * 4 + 3, 131 * 5 + 6, 123 * 4 + 23,
              true, true);
    return 0;
}
