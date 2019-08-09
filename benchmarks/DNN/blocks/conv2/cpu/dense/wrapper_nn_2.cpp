#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "wrapper_nn_2.h"
#include <tiramisu/utils.h>

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

int main(int, char**)
{
    Halide::Buffer<float> input(N+K, N+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K, K, FIn, FOut);
    Halide::Buffer<float> bias(FIn);
    Halide::Buffer<float> filter2(K, K, FIn, FOut);
    Halide::Buffer<float> bias2(FIn);
    Halide::Buffer<float> conv(N+K, N+K, FOut, BATCH_SIZE);
    Halide::Buffer<float> conv2_halide(N-K, N-K, FOut, BATCH_SIZE);
    Halide::Buffer<float> conv2_tiramisu(N, N, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);

    std::vector<double> duration_vector_1;
    std::vector<double> duration_vector_2;

    for (int y = 0; y < N+K; ++y)
        for (int x = 0; x < N+K; ++x)
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

     for (int z = 0; z < FIn; ++z)
        bias2(z) = 1;

     for (int y = 0; y < K; ++y)
        for (int x = 0; x < K; ++x)
            for (int z = 0; z < FIn; ++z)
		for (int q = 0; q < FOut; ++q)
		    filter2(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;

    for (int i=0; i<NB_TESTS; i++)
    {
	    double start1 = rtclock();
	    conv_halide(input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(), conv2_halide.raw_buffer());
	    double end1 = rtclock();
	    duration_vector_1.push_back((end1 - start1) * 1000);
    }

    std::cout << "\t\tHalide conv2" << ": " << median(duration_vector_1) << "; " << std::endl;
    std::cout << "\t\tResult" << ": ";

    int count = 0;
    for (int y = 0; y < N-K; ++y)
        for (int x = 0; x < N-K; ++x)
            for (int z = 0; z < FOut; ++z)
	        for (int n = 0; n < BATCH_SIZE; ++n)
		    if (count < 10)
		    {
			std::cerr << conv2_halide(x, y, z, n) << ", ";
			count++;
		    }
    std::cout << std::endl;

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = K;
    parameters(2) = FIn;
    parameters(3) = FOut;
    parameters(4) = BATCH_SIZE;


    for (int i=0; i<NB_TESTS; i++)
    {
        double start2 = rtclock();
        conv_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(), conv2_tiramisu.raw_buffer());
        double end2 = rtclock();
        duration_vector_2.push_back((end2 - start2) * 1000);
    }

    std::cout << "\t\tTiramisu conv2" << ": " << median(duration_vector_2) << "; " << std::endl;
    std::cout << "\t\tIntermediate Result" << ": ";
    count = 0;
    for (int y = 0; y < N-K; ++y)
        for (int x = 0; x < N-K; ++x)
            for (int z = 0; z < FIn; ++z)
	        for (int n = 0; n < BATCH_SIZE; ++n)
		    if (count < 10)
		    {
			std::cerr << conv(x, y, z, n) << ", ";
			count++;
		    }
    std::cout << std::endl;

    std::cout << "\t\tResult" << ": ";
    count = 0;
    for (int y = 0; y < N-K; ++y)
        for (int x = 0; x < N-K; ++x)
            for (int z = 0; z < FIn; ++z)
	        for (int n = 0; n < BATCH_SIZE; ++n)
		    if (count < 10)
		    {
			std::cerr << conv2_tiramisu(x, y, z, n) << ", ";
			count++;
		    }
    std::cout << std::endl;

    compare_4D_buffers("comparing Tiramisu output with Halide output", conv2_tiramisu, conv2_halide, 5);

    return 0;
}
