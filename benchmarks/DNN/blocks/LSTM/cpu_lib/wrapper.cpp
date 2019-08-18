/*
 * You need to execute the MKL configuration script before
 * compiling this benchmark.
 * For example, if it is located in /opt/intel/mkl/bin/mklvars.sh and you are in 64-bits :
 * source /opt/intel/mkl/bin/mklvars.sh intel64
 */

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#include "configure.h"
#include "lstm.o.h"

int main(int argc, char *argv[])
{   
    int warmupN = 10;
    if (argc > 1)
        warmupN = atoi(argv[1]);

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> buf_Weights(4 * FEATURE_SIZE, FEATURE_SIZE, 2, NUM_LAYERS);
    Halide::Buffer<DATA_TYPE> buf_biases(4 * FEATURE_SIZE, NUM_LAYERS);

    Halide::Buffer<DATA_TYPE> buf_input(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<DATA_TYPE> buf_output(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);

    Halide::Buffer<DATA_TYPE> buf_h(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH + 1, NUM_LAYERS + 1);
    Halide::Buffer<DATA_TYPE> buf_c(FEATURE_SIZE, BATCH_SIZE, NUM_LAYERS);

    // Initialize weights
    std::srand(0);

    for (int i = 0; i < NUM_LAYERS; i++)
        for (int l = 0; l < FEATURE_SIZE; l++)
            for (int k = 0; k < 4 * FEATURE_SIZE; k++)      
                buf_Weights(k, l, 0, i) = ((float)(rand()%256 - 128)) / 1270.f;

    for (int i = 0; i < NUM_LAYERS; i++)
        for (int l = 0; l < FEATURE_SIZE; l++)
            for (int k = 0; k < 4 * FEATURE_SIZE; k++)
                buf_Weights(k, l, 1, i) = ((float)(rand()%256 - 128)) / 1270.f;

    for (int i = 0; i < NUM_LAYERS; i++)
        for (int j = 0; j < 4 * FEATURE_SIZE; j++)
            buf_biases(j, i) = ((float)(rand()%256 - 128)) / 1270.f;

    for (int i = 0; i < SEQ_LENGTH; i++)
        for (int j = 0; j < BATCH_SIZE; j++)
            for (int k = 0; k < FEATURE_SIZE; k++)
                buf_input(k, j, i) = ((float)(rand()%256 - 128)) / 1270.f;

    std::cout << "Initalization done" << std::endl;

    // Warmup
    for (int i = 0; i < warmupN; i++) {
        lstm(
            buf_Weights.raw_buffer(),
            buf_biases.raw_buffer(),
            buf_input.raw_buffer(),
            buf_h.raw_buffer(),
            buf_c.raw_buffer(),
            buf_output.raw_buffer()
        );
    }

    std::cout << "Warmup done" << std::endl;

    // Execute Tiramisu code
    std::vector<double> durations;
    for (int i = 0; i < NB_TESTS; i++) {
        double start = rtclock();

        lstm(
            buf_Weights.raw_buffer(),
            buf_biases.raw_buffer(),
            buf_input.raw_buffer(),
            buf_h.raw_buffer(),
            buf_c.raw_buffer(),
            buf_output.raw_buffer()
        );

        double end = rtclock();
        durations.push_back((end - start) * 1000);
    }

    std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;

    std::cout << "LSTM done" << std::endl;
   
    // Write results to file
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < SEQ_LENGTH; ++n)
        for (int z = 0; z < BATCH_SIZE; ++z)
            for (int y = 0; y < FEATURE_SIZE; ++y)
                resultfile << std::setprecision(10) << buf_output(y, z, n) << std::endl;

    resultfile.close();

    std::cout << "\t\t Result"
              << ":\n\n";

    // Check for correctness with MKLDNN
    std::ifstream infile1("tiramisu_result.txt"), infile2("mkldnn_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;

    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);
        
        if (std::abs(f1 - f2) < 0.0001)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

    return 0;
}
