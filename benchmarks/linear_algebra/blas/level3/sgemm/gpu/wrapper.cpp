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

    std::cout << std::endl << "----------" << std::endl;
    std::cout << "Running GPU GEMM (TMM) benchmark: testN: " << testN
              << ", check correctness: " << check_correctness
              << ", (M,N,K): (" << M << "," << N << "," << K << ")" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;
    
    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(N * K * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices with random values:
    for (int i = 0; i < M * K; i++) A[i] = std::rand() % 100;
    for (int i = 0; i < N * K; i++) B[i] = std::rand() % 100;
    for (int i = 0; i < M * N; i++) C[i] = std::rand() % 100;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {K, M});
    Halide::Buffer<DATA_TYPE> B_buf(B, {K, N});
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

        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0;
                for (int k = 0; k < K; k++) {
                    // Note that indices are flipped (see tutorial 2)
                    acc += A_buf(k, i) * B_buf(k, j);
                }
                C2_buf(j, i) = acc * alpha + C[i * N + j] * beta;
            }
        }
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "CPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << "ms" << std::endl << std::flush;

        compare_buffers("matmul", C_buf, C2_buf);
    }

    // GPU Multiplication

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

    for (int i = 0; i < testN + (check_correctness ? 1 : 0); i++) {
        float *d_A;
        float *d_B;
        float *d_C;
        cudaMalloc((void**)&d_A, M * K * sizeof(*A));
        cudaMalloc((void**)&d_B, N * K * sizeof(*A));
        cudaMalloc((void**)&d_C, M * N * sizeof(*A));

        cublasSetMatrix(K, M, sizeof(*A), A, K, d_A, K);
        cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K);
        cublasSetMatrix(N, M, sizeof(*C), C, N, d_C, N);

        float alpha_var = alpha;
        float beta_var = beta;

        // Since cublas is column-major we swap A and B to get C as row-major result
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_var, d_B, K, d_A, K, &beta_var, d_C, N);

        cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "cublas matmul done (excluding cublasHandle creation): "
              << (std::chrono::duration<double,std::milli>(t2 - t1) / testN).count() << "ms" << std::endl << std::flush;

    cublasDestroy(handle);

    if (check_correctness) {
        std::cout << "Checking cublas result:" << std::endl;
        bool flag = true;
        for (int i = 0; i < M && flag; i++) {
            for (int j = 0; j < N; j++) {
                if (C2_buf(j, i) != C[i * N + j]) {
                    std::cout << "cublas-validation difference!:" << std::endl;
                    std::cout << i << " " << j << " " << C[i * N + j] << " " << C2_buf(j, i) << std::endl;
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            std::cout << "cublas and validation matches." << std::endl;
        }
    }

    bool print_matrices = false;
    if (print_matrices) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << C2_buf(i, j) << " ";
            }
            std::cout << std::endl;
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    free(A);
    free(B);
    free(C);

    std::cout << "----------" << std::endl << std::endl;

    return 0;
}
