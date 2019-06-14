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
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;

    Halide::Buffer<float> input(Z_BLOCKING, N + 2, N + 2, Z_NB_BLOCKS, BATCH_SIZE);
    Halide::Buffer<float> conv_filter(Z_BLOCKING, FOUT_BLOCKING, K_X, K_Y, FOUT_NB_BLOCKS, Z_NB_BLOCKS);
    Halide::Buffer<float> conv_bias(GR);

    Halide::Buffer<float> output(FOUT_BLOCKING, N, N, FOUT_NB_BLOCKS, BATCH_SIZE);

    // Initialize buffers
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int j = 0; j < N + 2; ++j) {
                input(z%Z_BLOCKING, 0, j, z/Z_BLOCKING, n) = 0.f;
                input(z%Z_BLOCKING, j, 0, z/Z_BLOCKING, n) = 0.f;
                input(z%Z_BLOCKING, N + 1, j, z/Z_BLOCKING, n) = 0.f;
                input(z%Z_BLOCKING, j, N + 1, z/Z_BLOCKING, n) = 0.f;
            }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int y = 1; y <= N; ++y)
                for (int x = 1; x <= N; ++x)
                    input(z%Z_BLOCKING, x, y, z/Z_BLOCKING, n) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        for (int z = 0; z < 4*GR; ++z)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter(z%Z_BLOCKING, fout%FOUT_BLOCKING, k_x, k_y, fout/FOUT_BLOCKING, z/Z_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias(fout) = ((float)(rand()%256 - 128)) / 127.f;

    std::cout << "\t\tBuffers initialized" << std::endl;

    for (int i = 0; i < NB_TESTS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        densenet_block(input.raw_buffer(), conv_filter.raw_buffer(), conv_bias.raw_buffer(), output.raw_buffer());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        duration_vector.push_back(duration);
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
    int nb_errors = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < GR; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    mkl_result >> tmp;
                    if (abs(output(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) > 0.01)
                        nb_errors++;
                }

    std::cout << "\t\tResult"
              << ":\n\n";

    cout << "\t\tPercentage of correctness " << 100.f - ((double)nb_errors)/BATCH_SIZE*GR*N*N << "%" << endl << endl;

    return 0;
}