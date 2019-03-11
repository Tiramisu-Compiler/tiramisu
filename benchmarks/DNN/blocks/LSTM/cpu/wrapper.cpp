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

#include "wrapper.h"

typedef std::chrono::duration<double,std::milli> duration_t;

int main(int argc, char *argv[])
{
    int warmupN = 0;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<int32_t> buf_params(4);
    Halide::Buffer<float> buf_Weights(FEATURE_SIZE, 4 * FEATURE_SIZE, 2, NUM_LAYERS);
    Halide::Buffer<float> buf_biases(4 * FEATURE_SIZE, NUM_LAYERS);
    Halide::Buffer<float> buf_x(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<float> buf_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);

    buf_params(0) = FEATURE_SIZE;
    buf_params(1) = BATCH_SIZE;
    buf_params(2) = NUM_LAYERS;
    buf_params(3) = SEQ_LENGTH;

    std::srand(0);
    for (int i = 0; i < NUM_LAYERS; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4 * FEATURE_SIZE; k++) {
                for (int l = 0; l < FEATURE_SIZE; l++) {
                    buf_Weights(l, k, j, i) = (std::rand() % 200 - 100) / 100.;
                }
            }
        }
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        for (int j = 0; j < 4 * FEATURE_SIZE; j++) {
            buf_biases(j, i) = (std::rand() % 200 - 100) / 100.;
        }
    }
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < FEATURE_SIZE; k++) {
                buf_x(k, j, i) = (std::rand() % 200 - 100) / 100.;
            }
        }
    }

    for (int i = 0; i < warmupN; i++) {
        lstm(buf_params.raw_buffer(),
             buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
    }

    std::vector<duration_t> durations;
    for (int i = 0; i < NB_TESTS; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        lstm(buf_params.raw_buffer(),
             buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
        auto t2 = std::chrono::high_resolution_clock::now();
        durations.push_back(t2 - t1);
    }

    if (NB_TESTS > 0) {
        std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;
    }

    std::cout << "LSTM done" << std::endl;
   
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < SEQ_LENGTH; ++n)
        for (int z = 0; z < BATCH_SIZE; ++z)
            for (int y = 0; y < FEATURE_SIZE; ++y)
            {
                resultfile << buf_y(y, z, n);
                resultfile << "\n";
            }
    resultfile.close();

    std::cout << "\t\t Result"
              << ":\n\n";

    std::ifstream infile1("tiramisu_result.txt"), infile2("mkldnn_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;

    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);
        if (abs(f1 - f2) < 0.02)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

    return 0;
}
