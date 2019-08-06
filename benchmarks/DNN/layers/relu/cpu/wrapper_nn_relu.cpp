#include "relu_layer_generator_tiramisu.o.h"
#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <string>
#include <time.h>
#include <iostream>
#include "configure.h"
using namespace std;

int main(int, char **)
{
    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> output(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> parameters(4);
    std::vector<double> duration_vector;

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
        double start = rtclock();
        relu_tiramisu(input.raw_buffer(), parameters.raw_buffer(), output.raw_buffer());

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }
    std::cout << "\t\tTiramisu relu duration"
              << ": " << median(duration_vector) << "; " << std::endl;

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
