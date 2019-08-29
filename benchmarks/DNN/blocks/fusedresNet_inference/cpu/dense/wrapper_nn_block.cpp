#define __TIRAMISU_WRAPPER__
#include <iostream>
#include <cstdlib>
#include <Halide.h>
#include <chrono>
#include <tiramisu/tiramisu.h>

#include "fused_resnet_block_generator_tiramisu.o.h"
#include "configure.h"

using namespace std;

int main()
{
    srand(1);
    std::vector<double> duration_vector;

    Halide::Buffer<float> input(FIN_BLOCKING, N+2, N+2, FIN_NB_BLOCKS, BATCH_SIZE);

    Halide::Buffer<float> bn1_scale(FOut), bn1_shift(FOut);
    Halide::Buffer<float> bn1_mean(FOut), bn1_variance(FOut);

    Halide::Buffer<float> filter1(FOUT_BLOCKING, FIN_BLOCKING, K_X, K_Y, FIN_NB_BLOCKS, FOUT_NB_BLOCKS);
    Halide::Buffer<float> bias1(FOut);

    Halide::Buffer<float> bn2_scale(FOut), bn2_shift(FOut);
    Halide::Buffer<float> bn2_mean(FOut), bn2_variance(FOut);

    Halide::Buffer<float> filter2(FOUT_BLOCKING, FIN_BLOCKING, K_X, K_Y, FIN_NB_BLOCKS, FOUT_NB_BLOCKS);
    Halide::Buffer<float> bias2(FOut);

    Halide::Buffer<float> conv1_buf(FOUT_BLOCKING, N+2, N+2, FOUT_NB_BLOCKS, BATCH_SIZE);
    Halide::Buffer<float> conv2_buf(FOUT_BLOCKING, N, N, FOUT_NB_BLOCKS, BATCH_SIZE);

    // Initialize buffers
    for (int fout = 0; fout < FOut; ++fout) {
        bn1_scale(fout) = 1.f;
        bn1_shift(fout) = 0.f;

        bn1_mean(fout) = ((float)(rand()%256)) / 127.f;
        bn1_variance(fout) = ((float)(rand()%256)) / 127.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    filter1(fout%FOUT_BLOCKING, fin%FIN_BLOCKING, k_x, k_y, fin/FIN_BLOCKING, fout/FOUT_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        bias1(fout) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout) {
        bn2_scale(fout) = 1.f;
        bn2_shift(fout) = 0.f;

        bn2_mean(fout) = ((float)(rand()%256)) / 127.f;
        bn2_variance(fout) = ((float)(rand()%256)) / 127.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    filter2(fout%FOUT_BLOCKING, fin%FIN_BLOCKING, k_x, k_y, fin/FIN_BLOCKING, fout/FOUT_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        bias2(fout) = ((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < FIn; ++fin)
            for (int y = 0; y < N + 2; ++y)
                for (int x = 0; x < N + 2; ++x)
                    input(fin%FIN_BLOCKING, x, y, fin/FIN_BLOCKING, n) = ((float)(rand()%256 - 128)) / 127.f;
    if (!TUNE_PARAMETERS)
      std::cout << "\t\tBuffers initialized" << std::endl;

    // Execute Tiramisu code
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();
        fused_resnet_block(
            input.raw_buffer(),
            filter1.raw_buffer(),
            bias1.raw_buffer(),
            bn1_scale.raw_buffer(),
            bn1_shift.raw_buffer(),
            bn1_mean.raw_buffer(),
            bn1_variance.raw_buffer(),
            filter2.raw_buffer(),
            bias2.raw_buffer(),
            bn2_scale.raw_buffer(),
            bn2_shift.raw_buffer(),
            bn2_mean.raw_buffer(),
            bn2_variance.raw_buffer(),
            conv1_buf.raw_buffer(),
            conv2_buf.raw_buffer()
        );

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tTiramisu ResNet block duration"
              << ": " << median(duration_vector) << " ms;" << std::endl;

    if (!TUNE_PARAMETERS){
      // Write results to file
      FILE* f = fopen("tiramisu_result.txt", "w");
      if (f == NULL) {
          printf("Error creating mkl_result.txt.\n");
          return 0;
      }

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N; ++y)
                  for (int x = 0; x < N; ++x)
                      fprintf(f, "%.10g\n", conv2_buf(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n));

      fclose(f);

      // Compare results with Intel MKL
      std::ifstream mkl_result("mkl_result.txt");
      float tmp;
      float file_count = 0, corr = 0;

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N; ++y)
                  for (int x = 0; x < N; ++x) {
                      mkl_result >> tmp;

                      file_count++;
                      if (abs(conv2_buf(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) <= 0.001)
                          corr++;
                  }

      std::cout << "\t\tResult"
                << ":\n\n";

      cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << endl << endl;
    }

    return 0;
}
