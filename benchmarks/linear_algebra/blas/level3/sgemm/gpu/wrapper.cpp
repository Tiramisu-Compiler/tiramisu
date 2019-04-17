#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "wrapper.h"
#include "configuration.h"

typedef std::chrono::duration<double,std::milli> duration_t;

// A helper function to determine runtime within Tiramisu function
extern "C"
float get_time(int32_t dummy)
{
    static auto t0 = std::chrono::high_resolution_clock::now();
    return duration_t(std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char *argv[])
{
    int testN = 1;
    int warmupN = 15;
    bool check_correctness = false;

    if (argc > 1) {
        testN = atoi(argv[1]);
    }
    if (argc > 2) {
        check_correctness = atoi(argv[2]);
        if (check_correctness) {
            testN = 1;
        }
    }
    if (argc > 3) {
        warmupN = atoi(argv[3]);
    }

    std::cout << std::endl << "----------" << std::endl;
    std::cout << "Running GPU GEMM benchmark: testN: " << testN
              << ", check correctness: " << check_correctness
              << ", (M,N,K) = (" << M << "," << N << "," << K << ")" << std::endl;

    float *A = (float*) malloc(M * K * sizeof(float));
    float *B = (float*) malloc(K * N * sizeof(float));
    float *C = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices with random values. The range is small to make sure
    // correctness comparison does not fail due to floating precision errors.
    for (int i = 0; i < M * K; i++) A[i] = std::rand() % 10 - 5;
    for (int i = 0; i < K * N; i++) B[i] = std::rand() % 10 - 5;
    for (int i = 0; i < M * N; i++) C[i] = std::rand() % 10 - 5;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {K, M});
    Halide::Buffer<DATA_TYPE> B_buf(B, {N, K});
    Halide::Buffer<DATA_TYPE> C_buf(N, M);
    Halide::Buffer<DATA_TYPE> Consts_buf(2);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Note that indices are flipped (see tutorial 2)
            C_buf(j, i) = C[i * N + j];
        }
    }
    Consts_buf(0) = 3; // A "random" alpha
    Consts_buf(1) = 2; // Ditto beta

    Halide::Buffer<float> time_start(1);
    Halide::Buffer<float> time_end(1);

    // Warm up:
    if (!check_correctness) {
        // GPU side tends to be slow for first couple of runs
        std::cout << "Warm up..." << std::endl;
        for (int i = 0; i < warmupN; i++) {
            matmul(Consts_buf.raw_buffer(),
                   A_buf.raw_buffer(),
                   B_buf.raw_buffer(),
                   C_buf.raw_buffer(),
                   time_start.raw_buffer(),
                   time_end.raw_buffer());
        }
        std::cout << "Warm up done" << std::endl;
    }

    // GPU Multiplication

    cudaProfilerStart();

    std::vector<duration_t> durations1;
    for (int i = 0; i < testN; i++) {
        matmul(Consts_buf.raw_buffer(),
               A_buf.raw_buffer(),
               B_buf.raw_buffer(),
               C_buf.raw_buffer(),
               time_start.raw_buffer(),
               time_end.raw_buffer());
        durations1.push_back(duration_t(time_end(0) - time_start(0)));
    }

    if (testN > 0) {
        std::cout << "GPU matmul done: " << median(durations1) << "ms" << std::endl << std::flush;
    }

    // CUBLAS SGEMM

    // Excluding handle creation which is time consuming
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<duration_t> durations2;
    for (int i = 0; i < testN; i++) {
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

        auto t1 = std::chrono::high_resolution_clock::now();
        // Since cublas is column-major we swap A and B to get C as row-major result
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_var, d_B, N, d_A, K, &beta_var, d_C, N);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        durations2.push_back(t2 - t1);

        cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    if (testN > 0) {
        std::cout << "cuBLAS matmul done: "
                  << median(durations2) << "ms" << std::endl << std::flush;
    }

    cublasDestroy(handle);

    if (check_correctness) {
        std::cout << "Comparing cuBLAS-Tiramisu outputs:" << std::endl;
        // A simple sanity check to make sure we are not comparing trivial matrices
        if (C[0] == 0 && C[1] == 0 && C[2] == 0) {
            std::cout << "Make sure outputs are not all zero!" << std::endl;
        }
        bool flag = true;
        for (int i = 0; i < M && flag; i++) {
            for (int j = 0; j < N; j++) {
                if (C_buf(j, i) != C[i * N + j]) {
                    std::cout << "Difference!:" << std::endl;
                    std::cout << i << " " << j << " " << C[i * N + j] << " " << C_buf(j, i) << std::endl;
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            std::cout << "cuBLAS and Tiramisu outputs match" << std::endl;
        }
    }

    free(A);
    free(B);
    free(C);

    std::cout << "----------" << std::endl << std::endl;

    return 0;
}
