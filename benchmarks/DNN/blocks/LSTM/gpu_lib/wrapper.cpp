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
    int testN_tiramisu = 100;
    int testN_cudnn = 100;
    int warmupN = 10;

    bool correctness_check = false;

    // Raw inputs
    float *raw_Weights = (float*) malloc(FEATURE_SIZE * 4 * FEATURE_SIZE * 2 * NUM_LAYERS * sizeof(float));
    float *raw_biases = (float*) malloc(4 * FEATURE_SIZE * NUM_LAYERS * sizeof(float));
    float *raw_x = (float*) malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(float));
    float *raw_y = (float*) malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(float));

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<float> buf_Weights(raw_Weights, {FEATURE_SIZE, 4 * FEATURE_SIZE, 2, NUM_LAYERS});
    Halide::Buffer<float> buf_biases(raw_biases, {4 * FEATURE_SIZE, NUM_LAYERS});
    Halide::Buffer<float> buf_x(raw_x, {FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH});
    Halide::Buffer<float> buf_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<float> buf_ref_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);

    if (correctness_check) {
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
        float max_err = 0;
        float max_rel_err = 0;
        std::cout << "Comparing against the reference:" << std::endl;
        for (int i = 0; i < SEQ_LENGTH; i++) {
            for (int j = 0; j < BATCH_SIZE; j++) {
                for (int k = 0; k < FEATURE_SIZE; k++) {
                    float res = buf_y(k, j, i);
                    float ref = buf_ref_y(k, j, i);
                    float err = std::abs(ref - res);
                    // Relative error:
                    float rel_err = err / std::max(std::abs(res), std::abs(ref));
                    max_err = std::max(max_err, err);
                    max_rel_err = std::max(max_rel_err, rel_err);
                    if (err > 0.01 && nn++ < 10) {
                        std::cout << i << " " << j << " " << k << ": "
                                  << res << " " << ref << ", "
                                  << err << " " << rel_err << std::endl;
                    }
                }
            }
        }

        std::cout << "Max error: " << max_err << std::endl;
        std::cout << "Max relative error: " << max_rel_err << std::endl;
    }

    setup_cudnn(SEQ_LENGTH, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE);

    for (int i = 0; i < warmupN; i++) {
        lstm(buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
        run_cudnn(raw_Weights, raw_biases, raw_x, raw_y);
    }

    std::cout << "warmup done" << std::endl;

    cudaProfilerStart();

    std::vector<duration_t> durations;
    for (int i = 0; i < testN_tiramisu; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        lstm(buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer());
        auto t2 = std::chrono::high_resolution_clock::now();
        durations.push_back(t2 - t1);
    }

    std::cout << "LSTM done" << std::endl;

    std::vector<duration_t> cudnn_durations;
    for (int i = 0; i < testN_cudnn; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        run_cudnn(raw_Weights, raw_biases, raw_x, raw_y);
        auto t2 = std::chrono::high_resolution_clock::now();
        cudnn_durations.push_back(t2 - t1);
    }

    free_cudnn();

    std::cout << "cudnn done" << std::endl;

    if (testN_tiramisu > 0) {
        std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;
    }
    if (testN_cudnn > 0) {
        std::cout << "cudnn median runtime: " << median(cudnn_durations) << "ms" << std::endl << std::flush;
    }

    return 0;
}
