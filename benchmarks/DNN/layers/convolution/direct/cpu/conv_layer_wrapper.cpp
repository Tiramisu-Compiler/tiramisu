#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "conv_layer_wrapper.h"
#include <tiramisu/utils.h>

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

int main(int, char**)
{
    int nb_sizes;
    int sizes[NB_MAX_SIZES][4];

    nb_sizes = fill_sizes_array(sizes, nb_sizes);

    for (int j = 0; j < nb_sizes; j++)
    {
        int C_N = sizes[j][0];
        int C_BATCH_SIZE = sizes[j][1];
        int C_FIn = sizes[j][2];
        int C_FOut = sizes[j][3];

		Halide::Buffer<float> input(C_N+K, C_N+K, C_FIn, C_BATCH_SIZE);
		Halide::Buffer<float> filter(K, K, C_FIn, C_FOut);
		Halide::Buffer<float> bias(C_FIn);
		Halide::Buffer<float> conv(C_N, C_N, C_FOut, C_BATCH_SIZE);
		Halide::Buffer<float> conv_tiramisu_buffer(C_N, C_N, C_FOut, C_BATCH_SIZE);
		Halide::Buffer<int> parameters(K);

		std::vector<double> duration_vector;

		for (int y = 0; y < C_N+K; ++y)
			for (int x = 0; x < C_N+K; ++x)
			for (int z = 0; z < C_FIn; ++z)
				for (int n = 0; n < C_BATCH_SIZE; ++n)
				input(x, y, z, n) = 1;

		for (int z = 0; z < C_FIn; ++z)
			bias(z) = 1;

		for (int y = 0; y < K; ++y)
			for (int x = 0; x < K; ++x)
			for (int z = 0; z < C_FIn; ++z)
				for (int q = 0; q < C_FOut; ++q)
				filter(x, y, z, q) = 1;

		std::cout << "\t\tBuffers initialized" << std::endl;

		unsigned int count = 0;

		// Initialize parameters[]
		parameters(0) = C_N;
		parameters(1) = K;
		parameters(2) = C_FIn;
		parameters(3) = C_FOut;
		parameters(4) = C_BATCH_SIZE;

		for (int i=0; i<NB_TESTS; i++)
		{
			double start = rtclock();
			conv_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv_tiramisu_buffer.raw_buffer());
			
			double end = rtclock();
			duration_vector.push_back((end - start) * 1000);
		}

		std::cout << "\t\tN = " << C_N << "; BATCH_SIZE = " << C_BATCH_SIZE << "; FIn = " << C_FIn << "; FOut = " << C_FOut << ";" << std::endl;
		std::cout << "\t\tTiramisu conv" << ": " << median(duration_vector) << "; " << std::endl;
		std::cout << "\t\tResult" << ": ";
		count = 0;
		for (int y = 0; y < C_N+K; ++y)
			for (int x = 0; x < C_N+K; ++x)
			for (int z = 0; z < C_FIn; ++z)
				for (int n = 0; n < C_BATCH_SIZE; ++n)
				if (count < 10)
				{
					std::cerr << conv_tiramisu_buffer(x, y, z, n) << ", ";
					count++;
				}
		std::cout << std::endl;
    }

    return 0;
}
