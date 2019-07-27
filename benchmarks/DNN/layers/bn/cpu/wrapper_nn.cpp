#include "bn_layer_generator_tiramisu.o.h"
#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include <tiramisu/utils.h>
#include <iomanip>

using namespace std;

int main(int, char **)
{
    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> output(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<int> parameters(3);
    Halide::Buffer<float> mean(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> variance(N, N, FIn, BATCH_SIZE);

    std::vector<double> duration_vector;

    srand(1);
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = rand() % 100;
                    
    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = FIn;
    parameters(2) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; i++)
    {
        double start = rtclock();
        bn_tiramisu(input.raw_buffer(), parameters.raw_buffer(), mean.raw_buffer(), variance.raw_buffer(), output.raw_buffer());
        
        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tTiramisu BN duration"
              << ": " << median(duration_vector) << "; " << std::endl;
              
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    resultfile << setprecision(10) << output(x, y, z, n) << std::endl;

    resultfile.close();

    std::cout << "\t\t Result"
              << ":\n\n";

    std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;
    
    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);

        if (abs(f1 - f2) <= 0.0001)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);
    return 0;
}
