#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

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
    std::cout << "Running GPU GEMM benchmark: testN: " << testN
              << ", check correctness: " << check_correctness << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;
    
    float *A = (float*) malloc(N * N * sizeof(float));
    float *B = (float*) malloc(N * N * sizeof(float));
    float *C = (float*) malloc(N * N * sizeof(float));

    // Initialize matrices with random values:
    for (int i = 0; i < N * N; i++) A[i] = std::rand() % 100;
    for (int i = 0; i < N * N; i++) B[i] = std::rand() % 100;
    for (int i = 0; i < N * N; i++) C[i] = std::rand() % 100;

    std::cout << "Buffers initialized" << std::endl << std::flush;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> A_buf(A, {N, N});
    Halide::Buffer<DATA_TYPE> B_buf(B, {N, N});
    Halide::Buffer<DATA_TYPE> C_buf(C, {N, N});
    Halide::Buffer<DATA_TYPE> O_buf(N, N);

    // Make a dummy call to set up GPU (initalization takes time)
    matmul(A_buf.raw_buffer(), B_buf.raw_buffer(), C_buf.raw_buffer(), O_buf.raw_buffer());

    // CPU Multiplication for correctness check

    if (check_correctness) {
        // Reference matrix multiplication

        Halide::Buffer<DATA_TYPE> O_val_buf(N, N);
        Halide::Buffer<DATA_TYPE> T_val_buf(N, N);
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // Note that indices are flipped (see tutorial 2)
                O_val_buf(j, i) = 0;
                T_val_buf(j, i) = 0;
            }
        }
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    // Note that indices are flipped (see tutorial 2)
                    T_val_buf(j, i) += A_buf(k, i) * B_buf(j, k);
                }
            }
        }
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    // Note that indices are flipped (see tutorial 2)
                    O_val_buf(j, i) += T_val_buf(k, i) * C_buf(j, k);
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

    free(A);
    free(B);
    free(C);

    std::cout << "----------" << std::endl << std::endl;

    return 0;
}
