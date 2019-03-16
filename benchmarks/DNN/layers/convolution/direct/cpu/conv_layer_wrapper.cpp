#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "conv_layer_wrapper.h"
#include <tiramisu/utils.h>

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

int main(int, char **)
{

    int nb_sizes;

    if (RUN_DIFFERENT_SIZES)
	nb_sizes = 27;
    else
	nb_sizes = 1;

    int sizes[nb_sizes][4];


    if (RUN_DIFFERENT_SIZES)
    {
	sizes[0][0] = 32;
	sizes[0][1] = 8;
	sizes[0][2] = 16;
	sizes[0][3] = 16;

	sizes[1][0] = 64;
	sizes[1][1] = 32;
	sizes[1][2] = 16;
	sizes[1][3] = 16;

	sizes[2][0] = 512;
	sizes[2][1] = 100;
	sizes[2][2] = 16;
	sizes[2][3] = 16;

	sizes[3][0] = 224;
	sizes[3][1] = 8;
	sizes[3][2] = 3;
	sizes[3][3] = 64;

	sizes[4][0] = 224;
	sizes[4][1] = 32;
	sizes[4][2] = 3;
	sizes[4][3] = 64;

	sizes[5][0] = 224;
	sizes[5][1] = 100;
	sizes[5][2] = 3;
	sizes[5][3] = 64;

	sizes[6][0] = 56;
	sizes[6][1] = 8;
	sizes[6][2] = 64;
	sizes[6][3] = 64;

	sizes[7][0] = 56;
	sizes[7][1] = 32;
	sizes[7][2] = 64;
	sizes[7][3] = 64;

	sizes[8][0] = 56;
	sizes[8][1] = 100;
	sizes[8][2] = 64;
	sizes[8][3] = 64;

	sizes[9][0] = 56;
	sizes[9][1] = 8;
	sizes[9][2] = 64;
	sizes[9][3] = 128;

	sizes[10][0] = 56;
	sizes[10][1] = 32;
	sizes[10][2] = 64;
	sizes[10][3] = 128;

	sizes[11][0] = 56;
	sizes[11][1] = 100;
	sizes[11][2] = 64;
	sizes[11][3] = 128;

	sizes[12][0] = 28;
	sizes[12][1] = 8;
	sizes[12][2] = 128;
	sizes[12][3] = 128;

	sizes[13][0] = 28;
	sizes[13][1] = 32;
	sizes[13][2] = 128;
	sizes[13][3] = 128;

	sizes[14][0] = 28;
	sizes[14][1] = 100;
	sizes[14][2] = 128;
	sizes[14][3] = 128;

	sizes[15][0] = 28;
	sizes[15][1] = 8;
	sizes[15][2] = 100;
	sizes[15][3] = 256;

	sizes[16][0] = 28;
	sizes[16][1] = 32;
	sizes[16][2] = 100;
	sizes[16][3] = 256;

	sizes[17][0] = 28;
	sizes[17][1] = 100;
	sizes[17][2] = 100;
	sizes[17][3] = 256;

	sizes[18][0] = 14;
	sizes[18][1] = 8;
	sizes[18][2] = 256;
	sizes[18][3] = 256;

	sizes[19][0] = 14;
	sizes[19][1] = 32;
	sizes[19][2] = 256;
	sizes[19][3] = 256;

	sizes[20][0] = 14;
	sizes[20][1] = 100;
	sizes[20][2] = 256;
	sizes[20][3] = 256;

	sizes[21][0] = 14;
	sizes[21][1] = 8;
	sizes[21][2] = 310;
	sizes[21][3] = 512;

	sizes[22][0] = 14;
	sizes[22][1] = 32;
	sizes[22][2] = 310;
	sizes[22][3] = 512;

	sizes[23][0] = 14;
	sizes[23][1] = 100;
	sizes[23][2] = 310;
	sizes[23][3] = 512;

	sizes[24][0] = 7;
	sizes[24][1] = 8;
	sizes[24][2] = 512;
	sizes[24][3] = 512;

	sizes[25][0] = 7;
	sizes[25][1] = 32;
	sizes[25][2] = 512;
	sizes[25][3] = 512;

	sizes[26][0] = 7;
	sizes[26][1] = 100;
	sizes[26][2] = 512;
	sizes[26][3] = 512;
    }
    else
    {
	sizes[0][0] = 512;
	sizes[0][1] = 100;
	sizes[0][2] = 16;
	sizes[0][3] = 16;
    }

    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int j = 0; j < nb_sizes; ++j)
    {

        int N = sizes[j][0];
        int BATCH_SIZE = sizes[j][1];
        int FIn = sizes[j][2];
        int FOut = sizes[j][3];

        Halide::Buffer<double> input(N + K, N + K, FIn, BATCH_SIZE);
        Halide::Buffer<double> filter(K, K, FIn, FOut);
        Halide::Buffer<double> bias(FIn);
        Halide::Buffer<double> conv(N, N, FOut, BATCH_SIZE);
        Halide::Buffer<double> conv_tiramisu_buffer(N, N, FOut, BATCH_SIZE);
        Halide::Buffer<int> parameters(5);

        std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
        std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

        for (int y = 0; y < N + K; ++y)
            for (int x = 0; x < N + K; ++x)
                for (int z = 0; z < FIn; ++z)
                    for (int n = 0; n < BATCH_SIZE; ++n)
                        input(x, y, z, n) = 1;

        for (int z = 0; z < FIn; ++z)
            bias(z) = 1;

        for (int y = 0; y < K; ++y)
            for (int x = 0; x < K; ++x)
                for (int z = 0; z < FIn; ++z)
                    for (int q = 0; q < FOut; ++q)
                        filter(x, y, z, q) = 1;

        std::cout << "\t\tBuffers initialized" << std::endl;

        unsigned int count = 0;

        // Initialize parameters[]
        parameters(0) = N;
        parameters(1) = K;
        parameters(2) = FIn;
        parameters(3) = FOut;
        parameters(4) = BATCH_SIZE;

        for (int i = 0; i < NB_TESTS; i++)
        {
            auto start1 = std::chrono::high_resolution_clock::now();
            conv_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv_tiramisu_buffer.raw_buffer());
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end1 - start1;
            duration_vector_2.push_back(duration);
        }

        std::cout << "\t\tTiramisu conv"
                  << ": " << median(duration_vector_2) << "; " << std::endl;

        resultfile << median(duration_vector_2) << "\n";

        std::cout << "\t\tResult"
                  << ": ";
        count = 0;
        for (int y = 0; y < N + K; ++y)
            for (int x = 0; x < N + K; ++x)
                for (int z = 0; z < FIn; ++z)
                    for (int n = 0; n < BATCH_SIZE; ++n)
                        if (count < 10)
                        {
                            std::cerr << conv_tiramisu_buffer(x, y, z, n) << ", ";
                            count++;
                        }
        std::cout << std::endl;
    }
    resultfile.close();
    return 0;
}
