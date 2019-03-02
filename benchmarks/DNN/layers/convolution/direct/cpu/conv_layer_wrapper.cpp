#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "conv_layer_wrapper.h"
#include <tiramisu/utils.h>
using namespace std;

int main(int, char**)
{
    Halide::Buffer<int> parameters(4);
    Halide::Buffer<int> strides(2);
    strides(0) = STRIDES;
    strides(1) = STRIDES;
    Halide::Buffer<int> padding(2);
    padding(0) = PADDING;
    padding(1) = PADDING;
    Halide::Buffer<int> kernel(2);
    kernel(0) = KERNEL;
    kernel(1) = KERNEL;

    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(KERNEL, KERNEL, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> inputPadd(N + 2 * PADDING, N + 2 * PADDING, FIn, BATCH_SIZE);
    Halide::Buffer<float> output((N - KERNEL + 2 * PADDING) / STRIDES + 1, (N - KERNEL + 2 * PADDING) / STRIDES + 1, FOut, BATCH_SIZE);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;
    srand(1);
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = rand() % 10;
    for (int z = 0; z < FOut; ++z)
        bias(z) = 0;

    for (int x = 0; x < KERNEL; ++x)
        for (int y = 0; y < KERNEL; ++y)
            for (int z = 0; z < FIn; ++z)
                for (int q = 0; q < FOut; ++q)
                    filter(x, y, z, q) = 1;
    std::cout << "\t\tBuffers initialized" << std::endl;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = FIn;
    parameters(2) = FOut;
    parameters(3) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        conv_tiramisu(parameters.raw_buffer(), filter.raw_buffer(),
                      bias.raw_buffer(), strides.raw_buffer(), padding.raw_buffer(),
                      kernel.raw_buffer(), input.raw_buffer(), inputPadd.raw_buffer(),
                      output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }
    std::cout << "\t\tTiramisu convolution duration"
              << ": " << median(duration_vector_2) << "; " << std::endl;
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z)
            for (int y = 0; y < (N - KERNEL + 2 * PADDING) / STRIDES + 1; ++y)
                for (int x = 0; x < (N - KERNEL + 2 * PADDING) / STRIDES + 1; ++x)
                    resultfile << output(x, y, z, n) << "\n";
    resultfile.close();

    std::cout << "\t\t Result"
              << ": ";

    std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt"); //infile2("mkldnn_result.txt")
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
