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

    Halide::Buffer<int> parameters(2);
    Halide::Buffer<double> input(N, N, 4*GR, BATCH_SIZE);
    Halide::Buffer<double> bn_scale(4*GR), bn_shift(4*GR);
    Halide::Buffer<double> conv_filter(K_X, K_Y, 4*GR, GR);
    Halide::Buffer<double> conv_bias(GR);

    Halide::Buffer<double> output(N, N, GR, BATCH_SIZE);

    // Initialize buffers
    for (int z = 0; z < 4*GR; ++z) {
        bn_scale(z) = ((double)(rand()%256)) / 255.f;
        if (bn_scale(z) == 0.f)
            bn_scale(z) = 1.f;

        bn_shift(z) = ((double)(rand()%256 - 128)) / 127.f;
    }

    for (int fout = 0; fout < GR; ++fout)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter(k_x, k_y, fin, fout) = ((double)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias(fout) = ((double)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input(x, y, z, n) = ((double)(rand() % 256)) / 255.f;

    std::cout << "\t\tBuffers initialized" << std::endl;

    parameters(0) = N;
    parameters(1) = BATCH_SIZE;

    for (int i = 0; i < NB_TESTS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        densenet_block(parameters.raw_buffer(), input.raw_buffer(), bn_scale.raw_buffer(), bn_shift.raw_buffer(), conv_filter.raw_buffer(), conv_bias.raw_buffer(), output.raw_buffer());
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
        for (int z = 0; z < GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.17g\n", output(x, y, z, n));

    fclose(f);

    // Compare results with Intel MKL
    std::ifstream mkl_result("mkl_result.txt");
    double tmp;
    int nb_errors = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    mkl_result >> tmp;
                    if (abs(output(x, y, z, n) - tmp) > 0.00000001)
                        nb_errors++;
                }

    std::cout << "\t\tResult"
              << ":\n\n";

    cout << "\t\tPercentage of correctness " << 100.f - ((double)nb_errors)/BATCH_SIZE*GR*N*N << "%" << endl << endl;

    return 0;
}