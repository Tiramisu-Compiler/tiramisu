#include <iostream>
#include <cstdlib>
#include <Halide.h>
#include <chrono>
#include <tiramisu/tiramisu.h>

#include "densenet_block_tiramisu.o.h"
#include "configure.h"

using namespace std;

int main()
{
    srand(1);
    std::vector<double> duration_vector;

    Halide::Buffer<float> input(FIN_BLOCKING, N+2, N+2, FIN_NB_BLOCKS, BATCH_SIZE);

    Halide::Buffer<float> bn_scale(4*GR), bn_shift(4*GR);
    Halide::Buffer<float> conv_filter(FOUT_BLOCKING, FIN_BLOCKING, K_X, K_Y, FOUT_NB_BLOCKS, FIN_NB_BLOCKS);
    Halide::Buffer<float> conv_bias(GR);

    Halide::Buffer<float> output(FOUT_BLOCKING, N, N, FOUT_NB_BLOCKS, BATCH_SIZE);

    Halide::Buffer<float> input_mean_buf(4*GR);
    Halide::Buffer<float> input_sd_buf(4*GR);
    Halide::Buffer<float> workspace_buf(FIN_BLOCKING, N + 2, N + 2, BATCH_SIZE);

    // Initialize buffers
    for (int fin = 0; fin < 4*GR; ++fin) {
        bn_scale(fin) = ((float)(rand()%256)) / 255.f;
        if (bn_scale(fin) == 0.f)
            bn_scale(fin) = 1.f;

        bn_shift(fin) = ((float)(rand()%256 - 128)) / 127.f;
    }

    for (int fout = 0; fout < GR; ++fout)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter(fout%FOUT_BLOCKING, fin%FIN_BLOCKING, k_x, k_y, fout/FOUT_BLOCKING, fin/FIN_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias(fout) = ((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int y = 0; y < N+2; ++y)
                for (int x = 0; x < N+2; ++x)
                    input(fin%FIN_BLOCKING, x, y, fin/FIN_BLOCKING, n) = ((float)(rand()%256 - 128)) / 127.f;

    std::cout << "\t\tBuffers initialized" << std::endl;

    // Execute Tiramisu code
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();
        densenet_block(
            input.raw_buffer(), 
            bn_scale.raw_buffer(), 
            bn_shift.raw_buffer(), 
            conv_filter.raw_buffer(), 
            conv_bias.raw_buffer(), 
            input_mean_buf.raw_buffer(),
            input_sd_buf.raw_buffer(),
            workspace_buf.raw_buffer(),
            output.raw_buffer()
        );
        
        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);	
    }

    std::cout << "\t\tTiramisu DenseNet block duration"
              << ": " << median(duration_vector) << " ms;" << std::endl;

    // Write results to file
    FILE* f = fopen("tiramisu_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 0;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < GR; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n));

    fclose(f);

    // Compare results with Intel MKL
    std::ifstream mkl_result("mkl_result.txt");
    float tmp;
    float file_count = 0, corr = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < GR; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    mkl_result >> tmp;

                    file_count++;
                    if (abs(output(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) <= 0.01)
                        corr++;
                }

    std::cout << "\t\tResult"
              << ":\n\n";

    cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << endl << endl;

    return 0;
}
