#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper.h"

typedef std::chrono::duration<double,std::milli> duration_t;

int main(int argc, char *argv[])
{
    int testN = 1;
    int warmupN = 0;

    int feature_size = 512;
    int batch_size = 64;
    int num_layers = 4;
    int seq_length = 100;

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<int32_t> buf_params(4);
    Halide::Buffer<float> buf_Weights(feature_size, 4 * feature_size, 2, num_layers);
    Halide::Buffer<float> buf_biases(4 * feature_size, num_layers);
    Halide::Buffer<float> buf_x(feature_size, batch_size, seq_length);
    Halide::Buffer<float> buf_y(feature_size, batch_size, seq_length);

    buf_params(0) = feature_size;
    buf_params(1) = batch_size;
    buf_params(2) = num_layers;
    buf_params(3) = seq_length;

    std::srand(0);
    for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4 * feature_size; k++) {
                for (int l = 0; l < feature_size; l++) {
                    buf_Weights(l, k, j, i) = (std::rand() % 200 - 100) / 100.;
                }
            }
        }
    }
    for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < 4 * feature_size; j++) {
            buf_biases(j, i) = (std::rand() % 200 - 100) / 100.;
        }
    }
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < batch_size; j++) {
            for (int k = 0; k < feature_size; k++) {
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
    for (int i = 0; i < testN; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        lstm(buf_params.raw_buffer(),
             buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
        auto t2 = std::chrono::high_resolution_clock::now();
        durations.push_back(t2 - t1);
    }

    if (testN > 0) {
        std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;
    }

    std::cout << "LSTM done" << std::endl;
   
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < seq_length; ++n)
        for (int z = 0; z < batch_size; ++z)
            for (int y = 0; y < feature_size; ++y)
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
