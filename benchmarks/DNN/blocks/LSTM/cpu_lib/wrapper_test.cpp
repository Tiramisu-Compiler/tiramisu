#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#include "configure.h"
#include "wrapper.h"
#include "mkl.h"

typedef std::chrono::duration<double,std::milli> duration_t;

// Helper function to record time from inside the function
extern "C"
float get_time(int32_t dummy)
{
    static auto t0 = std::chrono::high_resolution_clock::now();
    return duration_t(std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char *argv[])
{   
    int warmupN = 0;
    int check_correctness = 0;

    if (argc > 0) {
        warmupN = atoi(argv[0]);
    }
    if (argc > 1) {
        check_correctness = atoi(argv[1]);
    }

    int sizes[14][3];
    sizes[0][0] = 64;	sizes[0][1] = 64;	sizes[0][2] = 10;
    sizes[1][0] = 32;	sizes[1][1] = 64;	sizes[1][2] = 10;
    sizes[2][0] = 8;	sizes[2][1] = 64;	sizes[2][2] = 10;
    sizes[3][0] = 16;	sizes[3][1] = 64;	sizes[3][2] = 10;
    sizes[4][0] = 4;	sizes[4][1] = 64;	sizes[4][2] = 10;

    sizes[5][0] = 16;	sizes[5][1] = 128;	sizes[5][2] = 10;
    sizes[6][0] = 16;	sizes[6][1] = 32;	sizes[6][2] = 10;
    sizes[7][0] = 16;	sizes[7][1] = 16;	sizes[7][2] = 10;

    sizes[8][0] = 128;	sizes[8][1] = 16;	sizes[8][2] = 10;
    sizes[9][0] = 64;	sizes[9][1] = 16;	sizes[9][2] = 10;

    sizes[10][0] = 16;	sizes[10][1] = 16;	sizes[10][2] = 20;
    sizes[11][0] = 16;	sizes[11][1] = 16;	sizes[11][2] = 100;
    sizes[12][0] = 16;	sizes[12][1] = 16;	sizes[12][2] = 500;
    sizes[13][0] = 16;	sizes[13][1] = 16;	sizes[13][2] = 1000;

    for (int j = 0; j < 14; j++)
	{
	    int F = sizes[j][0];
	    int B = sizes[j][1];
	    int S = sizes[j][2];
	    int N = 4;

        // Raw inputs
        DATA_TYPE *raw_Weights = (DATA_TYPE*) malloc(F * 4 * F * 2 * N * sizeof(DATA_TYPE));
        DATA_TYPE *raw_biases = (DATA_TYPE*) malloc(4 * F * N * sizeof(DATA_TYPE));
        DATA_TYPE *raw_x = (DATA_TYPE*) malloc(F * B * S * sizeof(DATA_TYPE));

        // Note that indices are flipped (see tutorial 2)
        Halide::Buffer<DATA_TYPE> buf_Weights(raw_Weights, {F, 4 * F, 2, N});
        Halide::Buffer<DATA_TYPE> buf_biases(raw_biases, {4 * F, N});
        Halide::Buffer<DATA_TYPE> buf_x(raw_x, {F, B, S});
        Halide::Buffer<DATA_TYPE> buf_y(F, B, S);
        Halide::Buffer<float> time_start(1);
        Halide::Buffer<float> time_end(1);

        // Initialize weights
        std::srand(0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4 * F; k++)
                    for (int l = 0; l < F; l++)
                        buf_Weights(l, k, j, i) = (std::rand() % 200) / 100.;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 4 * F; j++)
                buf_biases(j, i) = (std::rand() % 200) / 100.;
        for (int i = 0; i < S; i++)
            for (int j = 0; j < B; j++)
                for (int k = 0; k < F; k++)
                    buf_x(k, j, i) = (std::rand() % 200) / 100.;

        std::cout << "Initalization done" << std::endl;

        // Warmup
        for (int i = 0; i < warmupN; i++) {
            lstm(buf_Weights.raw_buffer(),
                buf_biases.raw_buffer(),
                buf_x.raw_buffer(),
                buf_y.raw_buffer(),
                time_start.raw_buffer(),
                time_end.raw_buffer());
        }
        std::cout << "Warmup done" << std::endl;

        std::vector<duration_t> durations;
        for (int i = 0; i < NB_TESTS; i++) {
            lstm(buf_Weights.raw_buffer(),
                buf_biases.raw_buffer(),
                buf_x.raw_buffer(),
                buf_y.raw_buffer(),
                time_start.raw_buffer(),
                time_end.raw_buffer());
            durations.push_back(duration_t(time_end(0) - time_start(0)));
        }

        if (NB_TESTS > 0) {
            print_time("performance_LSTM_CPU.csv", "LSTM",
                {"Tiramisu"},
                {median(durations)});        
        }

        std::ofstream resultfile;
        resultfile.open("tiramisu_result.txt");

        for (int n = 0; n < S; ++n)
            for (int z = 0; z < B; ++z)
                for (int y = 0; y < F; ++y)
                {
                    resultfile << buf_y(y, z, n);
                    resultfile << "\n";
                }
        resultfile.close();
        if (check_correctness){
            std::ifstream infile1("tiramisu_result.txt"), infile2("../benchmarks/DNN/blocks/LSTM/cpu_lib/mkldnn_result.txt");
            std::string line1, line2;
            float file_count = 0, corr = 0, f1, f2;

            while (std::getline(infile1, line1))
            {
                std::getline(infile2, line2);
                file_count += 1;
                f1 = std::stof(line1);
                f2 = std::stof(line2);
                
                if (std::abs(f1 - f2) < 0.02)
                    corr += 1;
            }
            printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);
        }
    }
    return 0;
}
