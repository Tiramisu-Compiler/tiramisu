#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "maxpool_layer_generator_tiramisu.o.h"
#include <tiramisu/utils.h>
using namespace std;

bool compareFiles(const std::string &p1, const std::string &p2)
{
    std::ifstream f1(p1, std::ifstream::binary | std::ifstream::ate);
    std::ifstream f2(p2, std::ifstream::binary | std::ifstream::ate);

    if (f1.fail() || f2.fail())
    {
        return false; //File problem
    }

    if (f1.tellg() != f2.tellg())
    {
        return false; //Size mismatch
    }

    //Seek back to beginning and use std::equal to compare contents
    f1.seekg(0, std::ifstream::beg);
    f2.seekg(0, std::ifstream::beg);
    return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                      std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(f2.rdbuf()));
}

int main(int, char **)
{
    Halide::Buffer<int> parameters(3);
    Halide::Buffer<int> strides(2);
    strides(0) = S_X;
    strides(1) = S_Y;
    Halide::Buffer<int> padding(2);
    padding(0) = P_X;
    padding(1) = P_Y;
    Halide::Buffer<int> kernel(2);
    kernel(0) = K_X;
    kernel(1) = K_Y;

    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> inputPadd(N + 2 * P_X, N + 2 * P_Y, FIn, BATCH_SIZE);
    Halide::Buffer<float> output((N - K_X + 2 * P_X) / S_X + 1, (N - K_Y + 2 * P_Y) / S_Y + 1, FIn, BATCH_SIZE);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

    srand(1);
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = rand() % 10;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = FIn;
    parameters(2) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        maxpool_tiramisu(parameters.raw_buffer(), input.raw_buffer(), output.raw_buffer(),
                         strides.raw_buffer(), padding.raw_buffer(), kernel.raw_buffer(), inputPadd.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }
    std::cout << "\t\tTiramisu maxpool duration"
              << ": " << median(duration_vector_2) << "; " << std::endl;

    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < (N - K_Y + 2 * P_Y) / S_Y + 1; ++y)
                for (int x = 0; x < (N - K_X + 2 * P_X) / S_X + 1; ++x)
                    resultfile << output(x, y, z, n);
    resultfile.close();

    std::cout << "\t\tResult"
              << ": \n";

    const std::string in1 = "tiramisu_result.txt";
    const std::string in2 = "mkl_result.txt";
    //const std::string in2 = "mkldnn_result.txt";

    if (compareFiles(in1, in2))
        printf("\t\t\tcorrect\n\n");
    else
        printf("\t\t\t error \n\n");
    printf(" \n\n");

    return 0;
}
