#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "configuration.h"
#include "wrapper.h"
#include "wrapper_cudnn.h"

typedef std::chrono::duration<double,std::milli> duration_t;

int main(int argc, char *argv[])
{
    int testN = 100;
    int warmupN = 20;

    // Raw inputs
    float *raw_Weights = (float*) malloc(FEATURE_SIZE * 4 * FEATURE_SIZE * 2 * NUM_LAYERS * sizeof(float));
    float *raw_biases = (float*) malloc(4 * FEATURE_SIZE * NUM_LAYERS * sizeof(float));
    float *raw_x = (float*) malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(float));

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<float> buf_Weights(raw_Weights, {FEATURE_SIZE, 4 * FEATURE_SIZE, 2, NUM_LAYERS});
    Halide::Buffer<float> buf_biases(raw_biases, {4 * FEATURE_SIZE, NUM_LAYERS});
    Halide::Buffer<float> buf_x(raw_x, {FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH});
    Halide::Buffer<float> buf_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<float> buf_ref_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);

    std::srand(0);
    for (int i = 0; i < NUM_LAYERS; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 4 * FEATURE_SIZE; k++)
                for (int l = 0; l < FEATURE_SIZE; l++)
                    buf_Weights(l, k, j, i) = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < NUM_LAYERS; i++)
        for (int j = 0; j < 4 * FEATURE_SIZE; j++)
            buf_biases(j, i) = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < SEQ_LENGTH; i++)
        for (int j = 0; j < BATCH_SIZE; j++)
            for (int k = 0; k < FEATURE_SIZE; k++)
                buf_x(k, j, i) = (std::rand() % 200 - 100) / 100.;

    lstm(buf_Weights.raw_buffer(),
         buf_biases.raw_buffer(),
         buf_x.raw_buffer(),
         buf_y.raw_buffer());

    lstm_ref(buf_Weights.raw_buffer(),
         buf_biases.raw_buffer(),
         buf_x.raw_buffer(),
         buf_ref_y.raw_buffer());

    int nn = 0;
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < FEATURE_SIZE; k++) {
                float res = buf_y(k, j, i);
                float ref = buf_ref_y(k, j, i);
                float err = std::abs(ref - res);
                // Relative error:
                float rel_err = err / std::max(std::abs(res), std::abs(ref));
                if (err > 0 && nn++ < 100) {
                    std::cout << i << " " << j << " " << k << ": "
                              << res << " " << ref << ", "
                              << err << " " << rel_err << std::endl;
                }
            }
        }
    }

    return 0;
    /*
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

    setup_cudnn(SEQ_LENGTH, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE, raw_Weights, raw_biases, raw_x);

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
    */
}
