#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "wrapper.h"
#include "wrapper_cudnn.h"

typedef std::chrono::duration<double,std::milli> duration_t;

int main(int argc, char *argv[])
{
    int testN = 100;
    int warmupN = 20;

    // Gemm requires these sizes to be constant
    int feature_size = 512;
    int batch_size = 64;
    int num_layers = 4;
    int seq_length = 10;

    // Raw inputs
    float *raw_Weights = (float*) malloc(feature_size * 4 * feature_size * 2 * num_layers * sizeof(float));
    float *raw_biases = (float*) malloc(4 * feature_size * num_layers * sizeof(float));
    float *raw_x = (float*) malloc(feature_size * batch_size * seq_length * sizeof(float));

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<int32_t> buf_params(4);
    Halide::Buffer<float> buf_Weights(raw_Weights, {feature_size, 4 * feature_size, 2, num_layers});
    Halide::Buffer<float> buf_biases(raw_biases, {4 * feature_size, num_layers});
    Halide::Buffer<float> buf_x(raw_x, {feature_size, batch_size, seq_length});
    Halide::Buffer<float> buf_y(feature_size, batch_size, seq_length);
    Halide::Buffer<float> buf_ref_y(feature_size, batch_size, seq_length);

    // Initialize inputs
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

    lstm(buf_params.raw_buffer(),
         buf_Weights.raw_buffer(),
         buf_biases.raw_buffer(),
         buf_x.raw_buffer(),
         buf_y.raw_buffer());

    lstm_ref(buf_params.raw_buffer(),
         buf_Weights.raw_buffer(),
         buf_biases.raw_buffer(),
         buf_x.raw_buffer(),
         buf_ref_y.raw_buffer());

    /* Todo: result mismatch
    int nn = 0;
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < batch_size; j++) {
            for (int k = 0; k < feature_size; k++) {
                if (buf_y(k, j, i) != buf_ref_y(k, j, i) && nn++ < 100) {
                    std::cout << i << " " << j << " " << k << " " << buf_y(k, j, i) << " " << buf_ref_y(k, j, i) << std::endl;
                }
            }
        }
    }
    */

    for (int i = 0; i < warmupN; i++) {
        lstm(buf_params.raw_buffer(),
             buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
    }

    cudaProfilerStart();

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

    std::cout << "LSTM done" << std::endl;

    setup_cudnn(seq_length, num_layers, batch_size, feature_size, raw_Weights, raw_biases, raw_x);

    std::vector<duration_t> cudnn_durations;
    for (int i = 0; i < testN; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        run_cudnn();
        auto t2 = std::chrono::high_resolution_clock::now();
        cudnn_durations.push_back(t2 - t1);
    }

    free_cudnn();

    std::cout << "cudnn done" << std::endl;

    if (testN > 0) {
        std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;
        std::cout << "cudnn median runtime: " << median(cudnn_durations) << "ms" << std::endl << std::flush;
    }

    return 0;
}
