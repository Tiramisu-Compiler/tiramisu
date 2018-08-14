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
    std::cout << "Running sequential MM benchmark: testN: " << testN
              << ", check correctness: " << check_correctness
              << ", size: (" << S0 << ", " << S1 << ", " << S2 << ", " << S3 << ")" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;
    
    float *A = (float*) malloc(S0 * S1 * sizeof(float));
    float *B = (float*) malloc(S1 * S2 * sizeof(float));
    float *C = (float*) malloc(S2 * S3 * sizeof(float));

    // Initialize matrices with random values:
    for (int i = 0; i < S0 * S1; i++) A[i] = std::rand() % 10;
    for (int i = 0; i < S1 * S2; i++) B[i] = std::rand() % 10;
    for (int i = 0; i < S2 * S3; i++) C[i] = std::rand() % 10;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {S1, S0});
    Halide::Buffer<DATA_TYPE> B_buf(B, {S2, S1});
    Halide::Buffer<DATA_TYPE> C_buf(C, {S3, S2});
    Halide::Buffer<DATA_TYPE> O_buf(S3, S0);

    // Make a dummy call to set up GPU (initalization takes time)
    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer(), O_buf.raw_buffer());

    // CPU Multiplication for correctness check

    if (check_correctness) {
        // Reference matrix multiplication

        std::cout << "Running CPU multiplication.." << std::endl;

        Halide::Buffer<DATA_TYPE> O_val_buf(S3, S0);
        Halide::Buffer<DATA_TYPE> T_val_buf(S2, S0);
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < S0; i++) {
            for (int k = 0; k < S2; k++) {
                // Note that indices are flipped (see tutorial 2)
                T_val_buf(k, i) = 0;
            }
        }
        for (int i = 0; i < S0; i++) {
            for (int l = 0; l < S3; l++) {
                // Note that indices are flipped (see tutorial 2)
                O_val_buf(l, i) = 0;
            }
        }
        for (int j = 0; j < S1; j++) {
            for (int i = 0; i < S0; i++) {
                for (int k = 0; k < S2; k++) {
                    // Note that indices are flipped (see tutorial 2)
                    T_val_buf(k, i) += A_buf(j, i) * B_buf(k, j);
                }
            }
        }
        for (int k = 0; k < S2; k++) {
            for (int i = 0; i < S0; i++) {
                for (int l = 0; l < S3; l++) {
                    // Note that indices are flipped (see tutorial 2)
                    O_val_buf(l, i) += T_val_buf(k, i) * C_buf(l, k);
                }
            }
        }
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "CPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() << "ms" << std::endl << std::flush;

        compare_buffers("matmul", O_buf, O_val_buf);
    }

    // GPU Multiplication

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < testN; i++) {
        matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer(), O_buf.raw_buffer());
    }
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "GPU matmul done: " << (std::chrono::duration<double,std::milli>(t2 - t1)).count() / testN << "ms" << std::endl << std::flush;

    // CUBLAS SGEMM

    // Transposed copies for cublas
    float *A_T = (float*) malloc(S0 * S1 * sizeof(float));
    float *B_T = (float*) malloc(S1 * S2 * sizeof(float));
    float *C_T = (float*) malloc(S2 * S3 * sizeof(float));
    float *O_T = (float*) malloc(S0 * S3 * sizeof(float));
    // Transpose
    for (int i = 0; i < S0; i++) for (int j = 0; j < S1; j++) A_T[i + j * S0] = A[i * S1 + j];
    for (int i = 0; i < S1; i++) for (int j = 0; j < S2; j++) B_T[i + j * S1] = B[i * S2 + j];
    for (int i = 0; i < S2; i++) for (int j = 0; j < S3; j++) C_T[i + j * S2] = C[i * S3 + j];

    // Excluding handle creation which is time consuming
    cublasHandle_t handle;
    cublasCreate(&handle);

    t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < testN; i++) {
        float *d_A;
        float *d_B;
        float *d_C;
        float *d_T;
        float *d_O;
        cudaMalloc((void**)&d_A, S0 * S1 * sizeof(*A));
        cudaMalloc((void**)&d_B, S1 * S2 * sizeof(*A));
        cudaMalloc((void**)&d_C, S2 * S3 * sizeof(*A));
        cudaMalloc((void**)&d_T, S0 * S2 * sizeof(*A));
        cudaMalloc((void**)&d_O, S0 * S3 * sizeof(*A));

        cublasSetMatrix(S0, S1, sizeof(*A), A_T, S0, d_A, S0);
        cublasSetMatrix(S1, S2, sizeof(*B), B_T, S1, d_B, S1);
        cublasSetMatrix(S2, S3, sizeof(*C), C_T, S2, d_C, S2);

        float alpha_var = 1;
        float beta_var = 0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S0, S2, S1, &alpha_var, d_A, S0, d_B, S1, &beta_var, d_T, S0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S0, S3, S2, &alpha_var, d_T, S0, d_C, S2, &beta_var, d_O, S0);

        cublasGetMatrix(S0, S3, sizeof(*C), d_O, S0, O_T, S0);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_T);
        cudaFree(d_O);
    }

    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "cublas matmul done (excluding cublasHandle creation): "
              << (std::chrono::duration<double,std::milli>(t2 - t1) / testN).count() << "ms" << std::endl << std::flush;

    cublasDestroy(handle);

    bool check_cublas_difference = false;
    if (check_cublas_difference) {
        bool flag = true;
        for (int i = 0; i < S0 && flag; i++) {
            for (int j = 0; j < S3; j++) {
                if (O_buf(j, i) != O_T[i + j * S0]) {
                    std::cout << "cublas validation mismatch:" << std::endl;
                    std::cout << i << " " << j << " " << O_T[i + j * S0] << " " << O_buf(j, i) << std::endl;
                }
            }
        }
        if (flag) {
            std::cout << "cublas and validation match" << std::endl;
        }
    }

    free(A);
    free(B);
    free(C);
    free(A_T);
    free(B_T);
    free(C_T);
    free(O_T);

    std::cout << "----------" << std::endl << std::endl;

    return 0;
}
