#define __TIRAMISU_WRAPPER__
#include <iostream>
#include <cstdlib>
#include <Halide.h>
#include <chrono>
#include <tiramisu/tiramisu.h>

#include "conv_relu_maxpool_tiramisu.o.h"
#include "configure.h"

using namespace std;

int main()
{
    srand(1);
    std::vector<double> duration_vector;

    Halide::Buffer<float> input(FIn, N + 2, N + 2, BATCH_SIZE);

    Halide::Buffer<float> conv_filter(FOUT_BLOCKING, FIn, K_X, K_Y, FOUT_NB_BLOCKS);
    Halide::Buffer<float> conv_bias(FOut);

    Halide::Buffer<float> output(FOUT_BLOCKING, N/2, N/2, FOUT_NB_BLOCKS, BATCH_SIZE);

    // Initialize buffers
    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter(fout%FOUT_BLOCKING, fin, k_x, k_y, fout/FOUT_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        conv_bias(fout) = ((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < FIn; ++fin)
            for (int y = 0; y < N + 2; ++y)
                for (int x = 0; x < N + 2; ++x)
                    input(fin, x, y, n) = ((float)(rand()%256 - 128)) / 127.f;
    if (!TUNE_PARAMETERS)
      std::cout << "\t\tBuffers initialized" << std::endl;

    // Execute Tiramisu code
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();
        conv_relu_maxpool_block(
            input.raw_buffer(),
            conv_filter.raw_buffer(),
            conv_bias.raw_buffer(),
            output.raw_buffer()
        );

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tTiramisu Conv-ReLU-MaxPool block duration"
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
              for (int y = 0; y < N/2; ++y)
                  for (int x = 0; x < N/2; ++x)
                      fprintf(f, "%.10g\n", output(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n));

      fclose(f);

      // Compare results with Intel MKL
      std::ifstream mkl_result("mkl_result.txt");
      float tmp;
      float file_count = 0, corr = 0;

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N/2; ++y)
                  for (int x = 0; x < N/2; ++x) {
                      mkl_result >> tmp;

                      file_count++;
                      if (abs(output(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) <= 0.0001)
                          corr++;
                  }

      std::cout << "\t\tResult"
                << ":\n\n";

      cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << endl << endl;
    }

    return 0;
}
