#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "configure.h"
#include "wrapper.h"
#include </opt/intel/compilers_and_libraries_2017.5.239/linux/mkl/include/mkl_dnn.h>

typedef std::chrono::duration<double,std::milli> duration_t;

// Helper function to record time from inside the function
extern "C"
float get_time(int32_t dummy)
{
    static auto t0 = std::chrono::high_resolution_clock::now();
    return duration_t(std::chrono::high_resolution_clock::now() - t0).count();
}


/*extern "C"
float call_mkl_sgemm()
{
    static auto t0 = std::chrono::high_resolution_clock::now();
    return duration_t(std::chrono::high_resolution_clock::now() - t0).count();
}
*/  


int main(int argc, char *argv[])
{   
    int check_correctness = 0;
    int testN_tiramisu = 100;
    int testN_mkl = 100;
    int warmupN = 10;

    if (argc > 1) {
        testN_tiramisu = atoi(argv[1]);
    }
    if (argc > 2) {
        testN_mkl = atoi(argv[2]);
    }
    if (argc > 3) {
        warmupN = atoi(argv[3]);
    }
    if (argc > 4) {
        check_correctness = atoi(argv[4]);
    }

    // Raw inputs
    DATA_TYPE *raw_Weights = (DATA_TYPE*) malloc(FEATURE_SIZE * 4 * FEATURE_SIZE * 2 * NUM_LAYERS * sizeof(DATA_TYPE));
    DATA_TYPE *raw_biases = (DATA_TYPE*) malloc(4 * FEATURE_SIZE * NUM_LAYERS * sizeof(DATA_TYPE));
    DATA_TYPE *raw_x = (DATA_TYPE*) malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(DATA_TYPE));
    DATA_TYPE *raw_y = (DATA_TYPE*) malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(DATA_TYPE));

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> buf_Weights(raw_Weights, {FEATURE_SIZE, 4 * FEATURE_SIZE, 2, NUM_LAYERS});
    Halide::Buffer<DATA_TYPE> buf_biases(raw_biases, {4 * FEATURE_SIZE, NUM_LAYERS});
    Halide::Buffer<DATA_TYPE> buf_x(raw_x, {FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH});
    Halide::Buffer<DATA_TYPE> buf_y(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<float> time_start(1);
    Halide::Buffer<float> time_end(1);

    // Initialize weights
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

    std::cout << "Initalization done" << std::endl;

    //setup_mkl(SEQ_LENGTH, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE);

    // Warmup
    for (int i = 0; i < warmupN; i++) {
        lstm(buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer(),
             time_start.raw_buffer(),
             time_end.raw_buffer());
        //run_mkl(raw_Weights, raw_biases, raw_x, raw_y);
    }

    std::cout << "Warmup done" << std::endl;

    //cudaProfilerStart();

    std::vector<duration_t> durations;
    for (int i = 0; i < testN_tiramisu; i++) {
        lstm(buf_Weights.raw_buffer(),
             buf_biases.raw_buffer(),
             buf_x.raw_buffer(),
             buf_y.raw_buffer(),
             time_start.raw_buffer(),
             time_end.raw_buffer());
        durations.push_back(duration_t(time_end(0) - time_start(0)));
    }

    std::cout << "LSTM done" << std::endl;

   //std::vector<duration_t> mkl_durations;
  /*   for (int i = 0; i < testN_mkl; i++) {
        DATA_TYPE t = run_mkl(raw_Weights, raw_biases, raw_x, raw_y);
        mkl_durations.push_back(duration_t(t));
    }

    free_mkl();*/

   // std::cout << "mkl done" << std::endl;

    if (!check_correctness) {
        if (testN_tiramisu > 0) {
            std::cout << "LSTM median runtime: " << median(durations) << "ms" << std::endl << std::flush;
        }
      /*  if (testN_mkl > 0) {
            std::cout << "mkl median runtime: " << median(mkl_durations) << "ms" << std::endl << std::flush;
        }*/
    }

    /*    if (check_correctness && testN_tiramisu > 0 && testN_mkl > 0) {
        std::cout << "Testing mkl-Tiramisu output difference" << std::endl;
        for (int i = 0; i < SEQ_LENGTH; i++) {
            DATA_TYPE max_error = 0;
            for (int j = 0; j < BATCH_SIZE; j++) {
                for (int k = 0; k < FEATURE_SIZE; k++) {
                    DATA_TYPE res = buf_y(k, j, i);
                    DATA_TYPE cuda_res = raw_y[(i * BATCH_SIZE + j) * FEATURE_SIZE + k];
                    DATA_TYPE error = res - cuda_res;
                    if (error > max_error) {
                        max_error = error;
                    }
                }
            }
            std::cout << "Sequence #" << i << " max difference: " << max_error << std::endl;
        }
    }*/

    return 0;
}
