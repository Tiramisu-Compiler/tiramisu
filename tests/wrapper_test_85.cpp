#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_85.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

template <typename T>
void multiply(Halide::Buffer<T> &A, Halide::Buffer<T> &B, Halide::Buffer<T> &res,
              int N, int D, int M)
{
    init_buffer(res, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < D; k++)
                res(j, i) += A(k, i) * B(j, k);
}

template <typename T>
void init_grad(Halide::Buffer<T> &A, int N, int M)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            A(i, j) = (T)(i + j);
}

int main(int, char **)
{
    Halide::Buffer<int32_t> A1(D1, N1), B1(M1, D1), O1(M1, N1), R1(M1, N1), P1(3);
    Halide::Buffer<int32_t> A2(D2, N2), B2(M2, D2), O2(M2, N2), R2(M2, N2), P2(3);

    init_grad(A1, D1, N1);
    init_grad(B1, M1, D1);
    P1(0) = N1;
    P1(1) = D1;
    P1(2) = M1;
    multiply(A1, B1, R1, N1, D1, M1);

    tiramisu_generated_code(P1.raw_buffer(),
                            A1.get()->transposed(0, 1).raw_buffer(),
                            B1.get()->transposed(0, 1).raw_buffer(),
                            O1.raw_buffer());

    compare_buffers(TEST_ID_STR "_small", O1, R1);


    init_grad(A2, D2, N2);
    init_grad(B2, M2, D2);
    P2(0) = N2;
    P2(1) = D2;
    P2(2) = M2;
    multiply(A2, B2, R2, N2, D2, M2);

    tiramisu_generated_code(P2.raw_buffer(),
                            A2.get()->transposed(0, 1).raw_buffer(),
                            B2.get()->transposed(0, 1).raw_buffer(),
                            O2.raw_buffer());

    compare_buffers(TEST_ID_STR "_large", O2, R2);

    return 0;
}
