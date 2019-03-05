#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <string>
#include <time.h>
#include <iostream>
#include "configure.h"
#include <cuda_profiler_api.h>
#include "wrapper_cudnn.h"
using namespace std;

int main(int, char **)
{
    float *raw_input = (float*) malloc(N * N * FIn * BATCH_SIZE * sizeof(float));
    float *raw_output = (float*) malloc( ((N - K_X + 2 * P_X) / S_X + 1)* ((N - K_Y + 2 * P_Y) / S_Y + 1)* FIn * BATCH_SIZE * sizeof(float));

    Halide::Buffer<float> input(raw_input, {N, N, FIn, BATCH_SIZE});
    Halide::Buffer<float> output(raw_output , {N, N, FIn, BATCH_SIZE});

    Halide::Buffer<float> parameters(4);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

    srand(1);
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = rand() % 10 - 5;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = FIn;
    parameters(2) = BATCH_SIZE;
    parameters(3) = NEGATIVE_SLOPES;

    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        relu_layer_gpu(input.raw_buffer(), parameters.raw_buffer(), output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }
    std::cout << "\t\tTiramisu relu duration"
              << ": " << median(duration_vector_2) << "; " << std::endl;

    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                {
                    resultfile << output(x, y, z, n);
                    resultfile << "\n";
                }
    resultfile.close();

    std::cout << "ReLU Tiramisu done" << std::endl;

    setup_cudnn(N, FIn, BATCH_SIZE, NEGATIVE_SLOPES, raw_input, raw_output);

    std::vector<duration_t> cudnn_durations;
    for (int i = 0; i < testN; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        run_cudnn();
        auto t2 = std::chrono::high_resolution_clock::now();
        cudnn_durations.push_back(t2 - t1);
    }

    free_cudnn();

    std::cout << "cudnn done" << std::endl;
    return 0;
}
