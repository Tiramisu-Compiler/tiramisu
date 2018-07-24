#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "wrapper.h"
#include "configuration.h"

int main(int argc, char *argv[])
{
    int testN = 1;
    bool check_correctness = false;
    if (argc > 1) {
        testN = atoi(argv[1]);
    }
    if (argc > 2) {
        check_correctness = atoi(argv[2]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;
    
    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(K * N * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices with random values:
    for (int i = 0; i < M * K; i++) A[i] = std::rand() % 100;
    for (int i = 0; i < K * N; i++) B[i] = std::rand() % 100;
    for (int i = 0; i < M * N; i++) C[i] = std::rand() % 100;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {K, M});
    Halide::Buffer<DATA_TYPE> B_buf(B, {N, K});
    Halide::Buffer<DATA_TYPE> C_buf(N, M);
    Halide::Buffer<DATA_TYPE> C2_buf(N, M);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Note that indices are flipped (see tutorial 2)
            C_buf(j, i) = C[i * N + j];
        }
    }

    // Make a dummy call to set up GPU (initalization takes time)
    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer());

    // CPU Multiplication for correctness check

    if (check_correctness) {
        // Reference matrix multiplication

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                // Note that indices are flipped (see tutorial 2)
                C2_buf(j, i) = C[i * N + j] * beta;
            }
        }
        t1 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    // Note that indices are flipped (see tutorial 2)
                    C2_buf(j, i) += A_buf(k, i) * B_buf(j, k) * alpha;
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "CPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << "ms" << std::endl << std::flush;

        compare_buffers("matmul", C_buf, C2_buf);
    }

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < testN; i++) {
        matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer());
    }
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "GPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() / testN << "ms" << std::endl << std::flush;

    // CUBLAS SGEMM

    // Excluding handle creation which is time consuming
    cublasHandle_t handle;
    cublasCreate(&handle);

    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < testN; i++) {
        float *d_A;
        float *d_B;
        float *d_C;
        cudaMalloc((void**)&d_A, M * K * sizeof(*A));
        cudaMalloc((void**)&d_B, K * N * sizeof(*A));
        cudaMalloc((void**)&d_C, M * N * sizeof(*A));

        cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M);
        cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K);
        cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M);

        float alpha_var = alpha;
        float beta_var = beta;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha_var, d_A, M, d_B, K, &beta_var, d_C, M);

        cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "cublas matmul done (excluding cublasHandle creation): "
              << (std::chrono::duration<double,std::milli>(t2 - t1) / testN).count() << "ms" << std::endl << std::flush;

    cublasDestroy(handle);

    free(A);
    free(B);
    free(C);

    return 0;
}
