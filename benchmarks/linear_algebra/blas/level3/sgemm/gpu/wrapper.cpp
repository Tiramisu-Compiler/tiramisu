#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "wrapper.h"
#include "configuration.h"

typedef std::chrono::duration<double,std::milli> t_duration;

int main(int argc, char *argv[])
{
    int testN = 1;
    bool check_correctness = false;
    if (argc > 1) {
        testN = atoi(argv[1]);
    }
    if (argc > 2) {
        check_correctness = atoi(argv[2]);
        if (check_correctness) {
            testN = 0;
        }
    }

    std::cout << std::endl << "----------" << std::endl;
    std::cout << "Running GPU GEMM benchmark: testN: " << testN
              << ", check correctness: " << check_correctness
              << ", (M,N,K) = (" << M << "," << N << "," << K << ")" << std::endl;

    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(K * N * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices with random values:
    for (int i = 0; i < M * K; i++) A[i] = std::rand() % 10;
    for (int i = 0; i < K * N; i++) B[i] = std::rand() % 10;
    for (int i = 0; i < M * N; i++) C[i] = std::rand() % 10;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {K, M});
    Halide::Buffer<DATA_TYPE> B_buf(B, {N, K});
    Halide::Buffer<DATA_TYPE> C_buf(N, M);
    Halide::Buffer<DATA_TYPE> C2_buf(N, M);
    Halide::Buffer<DATA_TYPE> Consts_buf(2);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Note that indices are flipped (see tutorial 2)
            C_buf(j, i) = C[i * N + j];
        }
    }
    Consts_buf(0) = 3; // A "random" alpha
    Consts_buf(1) = 2; // Ditto beta

    // CPU Multiplication for correctness check

    if (check_correctness) {
        // GPU multiplication:
        matmul(Consts_buf.raw_buffer(), A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer());

        // Reference matrix multiplication
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0;
                for (int k = 0; k < K; k++) {
                    // Note that indices are flipped (see tutorial 2)
                    acc += A_buf(k, i) * B_buf(j, k);
                }
                C2_buf(j, i) = acc * Consts_buf(0) + C[i * N + j] * Consts_buf(1);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << "CPU matmul done: " << (t_duration(t2 - t1)).count() << "ms" << std::endl << std::flush;

        compare_buffers("matmul", C_buf, C2_buf);
    }

    // Warm up:
    if (!check_correctness) {
        // GPU side tends to be slow for first couple of runs
        std::cout << "Warm up..." << std::endl;
        for (int i = 0; i < 15; i++) {
            matmul(Consts_buf.raw_buffer(), A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer());
        }
        std::cout << "Warm up done" << std::endl;
    }

    // GPU Multiplication

    cudaProfilerStart();

    std::vector<t_duration> durations1;
    for (int i = 0; i < testN; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul(Consts_buf.raw_buffer(), A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer());
        auto t2 = std::chrono::high_resolution_clock::now();
        durations1.push_back(t2 - t1);
    }

    if (testN > 0) {
        std::cout << "GPU matmul done: " << median(durations1) << "ms" << std::endl << std::flush;
    }

    // CUBLAS SGEMM

    // Excluding handle creation which is time consuming
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<t_duration> durations2;
    for (int i = 0; i < testN + (check_correctness ? 1 : 0); i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        float *d_A;
        float *d_B;
        float *d_C;
        cudaMalloc((void**)&d_A, M * K * sizeof(*A));
        cudaMalloc((void**)&d_B, N * K * sizeof(*A));
        cudaMalloc((void**)&d_C, M * N * sizeof(*A));

        cublasSetMatrix(K, M, sizeof(*A), A, K, d_A, K);
        cublasSetMatrix(N, K, sizeof(*B), B, N, d_B, N);
        cublasSetMatrix(N, M, sizeof(*C), C, N, d_C, N);

        float alpha_var = Consts_buf(0);
        float beta_var = Consts_buf(1);

        // Since cublas is column-major we swap A and B to get C as row-major result
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_var, d_B, N, d_A, K, &beta_var, d_C, N);

        cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        auto t2 = std::chrono::high_resolution_clock::now();
        durations2.push_back(t2 - t1);
    }

    if (testN > 0) {
        std::cout << "cublas matmul done (excluding cublasHandle creation): "
                  << median(durations2) << "ms" << std::endl << std::flush;
    }

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
