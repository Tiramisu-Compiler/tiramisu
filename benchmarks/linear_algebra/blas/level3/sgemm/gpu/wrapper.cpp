#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "wrapper.h"

#define NN 1024

#define M 1024
#define N 1024
#define K 1024

int main(int, char **)
{
    Halide::Buffer<uint8_t> A_buf(NN, NN);
    Halide::Buffer<uint8_t> B_buf(NN, NN);
    // Initialize matrices with pseudorandom values:
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            A_buf(j, i) = (i + 3) * (j + 1);
            B_buf(j, i) = (i + 1) * j + 2;
        }
    }
    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Output
    Halide::Buffer<uint8_t> C1_buf(NN, NN);

    auto t1 = std::chrono::high_resolution_clock::now();
    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "GPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << std::endl << std::flush;

    // Reference matrix multiplication
    Halide::Buffer<uint8_t> C2_buf(NN, NN);
    init_buffer(C2_buf, (uint8_t)0);

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            for (int k = 0; k < NN; k++) {
                // Note that indices are flipped (see tutorial 2)
                C2_buf(j, i) += A_buf(k, i) * B_buf(j, k);
            }
        }
    }
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "CPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << std::endl << std::flush;

    compare_buffers("matmul", C1_buf, C2_buf);

    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(K * N * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(*A));
    cudaMalloc((void**)&d_B, K * N * sizeof(*A));
    cudaMalloc((void**)&d_C, M * N * sizeof(*A));

    cublasHandle_t handle;
    cublasCreate(&handle);

    t1 = std::chrono::high_resolution_clock::now();

    cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M);
    cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K);
    cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M);

    float alpha = 1, beta = 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M);

    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "cublas matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << std::endl << std::flush;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    free(A);
    free(B);
    free(C);

    return 0;
}
