#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_93.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

Halide::Buffer<double> mttkrp(int I, int J, int K, int L,
                              Halide::Buffer<double> &B,
                              Halide::Buffer<double> &C,
                              Halide::Buffer<double> &D)
{
    Halide::Buffer<double> A(J, I);
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            double s = 0.0;
            for (int k = 0; k < K; k++)
            {
                double t = 0.0;
                for (int l = 0; l < L; l++)
                {
                    t += B(l, k, i) * D(j, l);
                }
                s += t * C(j, k);
            }
            A(j, i) = s;
        }
    }
    return A;
}

int main(int, char **)
{
    double B[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    double C[] = {-0.1, 0.2, -0.3, 0.4};
    double D[] = {0.1, -0.2, 0.3, -0.4};

    int I = 2, J = 2, K = 2, L = 2;
    int Bsize[] = {I, K, L}, Csize[] = {K, J}, Dsize[] = {L, J};

    Halide::Buffer<double> B_(Halide::Float(64), B, {L, K, I}),
                           C_(Halide::Float(64), C, {J, K}),
                           D_(Halide::Float(64), D, {J, L}),
                           A_tiramisu(Halide::Float(64), {J, I});
    Halide::Buffer<int64_t> I_(Halide::Int(64), &I, {1}),
                            J_(Halide::Int(64), &J, {1}),
                            K_(Halide::Int(64), &K, {1}),
                            L_(Halide::Int(64), &L, {1}),
                            Bsize_(Halide::Int(64), Bsize, {4}),
                            Csize_(Halide::Int(64), Csize, {3}),
                            Dsize_(Halide::Int(64), Dsize, {3});

    auto A_reference = mttkrp(I, J, K, L, B_, C_, D_);

    tiramisu_generated_code(I_.raw_buffer(), J_.raw_buffer(),
                            K_.raw_buffer(), L_.raw_buffer(),
                            Bsize_.raw_buffer(), B_.raw_buffer(),
                            Csize_.raw_buffer(), C_.raw_buffer(),
                            Dsize_.raw_buffer(), D_.raw_buffer(),
                            A_tiramisu.raw_buffer());

    compare_buffers(TEST_ID_STR, A_tiramisu, A_reference);

    return 0;
}

