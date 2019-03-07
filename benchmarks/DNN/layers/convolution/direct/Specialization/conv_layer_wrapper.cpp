#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "generated_conv_layer.o.h"
#include <tiramisu/utils.h>

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

int main(int, char **)
{
    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K, K, FIn, FOut);
    Halide::Buffer<float> bias(FIn);
    Halide::Buffer<float> conv(N - K + 1, N - K + 1, FOut, BATCH_SIZE);
    Halide::Buffer<float> conv_tiramisu_buffer(N - K + 1, N - K + 1, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
            for (int z = 0; z < FIn; ++z)
                for (int n = 0; n < BATCH_SIZE; ++n)
                    input(x, y, z, n) = 1;

    for (int z = 0; z < FOut; ++z)
        bias(z) = 1;

    for (int y = 0; y < K; ++y)
        for (int x = 0; x < K; ++x)
            for (int z = 0; z < FIn; ++z)
                for (int q = 0; q < FOut; ++q)
                    filter(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;

    unsigned int count = 0;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = K;
    parameters(2) = FIn;
    parameters(3) = FOut;
    parameters(4) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        conv_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv_tiramisu_buffer.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }

    std::cout << "\t\tTiramisu conv"
              << ": " << median(duration_vector_2) << "; " << std::endl;

    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z)
            for (int y = 0; y < N - K + 1; ++y)
                for (int x = 0; x < N - K + 1; ++x)
                    resultfile << conv_tiramisu_buffer(x, y, z, n) << "\n";

    resultfile.close();

    std::cout << "\t\t Result :\n\n";

    std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;

    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);

        if (abs(f1 - f2) == 0)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

    return 0;
}