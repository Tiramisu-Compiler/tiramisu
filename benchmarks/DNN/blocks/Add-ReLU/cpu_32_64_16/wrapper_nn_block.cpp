#define __TIRAMISU_WRAPPER__
#include <iostream>
#include <cstdlib>
#include <Halide.h>
#include <chrono>
#include <tiramisu/tiramisu.h>

#include "add_relu_inplace_32_64_16_tiramisu.o.h"
#include "configure.h"

using namespace std;

int main()
{
    srand(1);
    std::vector<double> duration_vector;

    Halide::Buffer<float> x_buf(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> y_buf(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> output(N, N, FIn, BATCH_SIZE);
    // Initialize buffers
    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int n_channels = 0; n_channels < FIn; ++n_channels)
        for (int y = 0; y < N; ++y)
          for (int x = 0; x < N; ++x)
            x_buf(x, y, n_channels, n) = 1.f;//((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int n_channels = 0; n_channels < FIn; ++n_channels)
        for (int y = 0; y < N; ++y)
          for (int x = 0; x < N; ++x)
            output(x, y, n_channels, n) = 2.f;//((float)(rand()%256 - 128)) / 127.f;

    std::cout << "\t\tBuffers initialized" << std::endl;
    // Execute Tiramisu code
    for (int i = 0; i < NB_TESTS; ++i) {
        for (int n = 0; n < BATCH_SIZE; ++n)
          for (int n_channels = 0; n_channels < FIn; ++n_channels)
            for (int y = 0; y < N; ++y)
              for (int x = 0; x < N; ++x)
                output(x, y, n_channels, n) = 2.f;
        double start = rtclock();
        add_relu_inplace_32_64_16_block(
            x_buf.raw_buffer(),
            output.raw_buffer()
        );

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }
    std::cout << "\t\tTiramisu Add-ReLU block duration"
              << ": " << median(duration_vector) << " ms;" << std::endl;

    // Write results to file
    FILE* f = fopen("tiramisu_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 0;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int n_channels = 0; n_channels < FIn; ++n_channels)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output(x, y, n_channels, n));

    fclose(f);

    // Compare results with Intel MKL
    std::ifstream mkl_result("mkl_result.txt");
    float tmp;
    float file_count = 0, corr = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int n_channels = 0; n_channels < FIn; ++n_channels)
        for (int y = 0; y < N; ++y)
          for (int x = 0; x < N; ++x) {
            mkl_result >> tmp;

            file_count++;
            if (abs(output(x, y, n_channels, n) - tmp) <= 0.0001)
                corr++;
          }

    std::cout << "\t\tResult"
              << ":\n\n";

    cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << endl << endl;


    return 0;
}
