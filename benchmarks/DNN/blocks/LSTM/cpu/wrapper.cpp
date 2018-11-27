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
    Halide::Buffer<float> buf_Weights(feature_size, feature_size, 8, num_layers);
    Halide::Buffer<float> buf_biases(feature_size, 4, num_layers);
    Halide::Buffer<float> buf_x(feature_size, batch_size, seq_length);
    Halide::Buffer<float> buf_y(feature_size, batch_size, seq_length);

    buf_params(0) = feature_size;
    buf_params(1) = batch_size;
    buf_params(2) = num_layers;
    buf_params(3) = seq_length;

    // TODO: Initialize inputs

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

    return 0;
}
