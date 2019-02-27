#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "generated_vgg_block.o.h"
#include <tiramisu/utils.h>
using namespace std;

int main(int, char **)
{
	Halide::Buffer<float> input(N + K, N + K, FIn, BATCH_SIZE);
	Halide::Buffer<float> filter(K + 1, K + 1, FIn, FOut);
	Halide::Buffer<float> bias(FOut);
	Halide::Buffer<float> conv(N, N, FOut, BATCH_SIZE);
	Halide::Buffer<float> filter2(K + 1, K + 1, FOut, FOut);
	Halide::Buffer<float> bias2(FOut);
	Halide::Buffer<float> conv2(N - K, N - K, FOut, BATCH_SIZE);
	Halide::Buffer<float> output(N - 2 * K, N - 2 * K, FOut, BATCH_SIZE);
	Halide::Buffer<int> parameters(5);

	std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
	std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;

	srand(1);
	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FIn; ++z)
			for (int y = 0; y < N + K; ++y)
				for (int x = 0; x < N + K; ++x)
					input(x, y, z, n) = (std::rand() % 200 - 100) / 100.;

	for (int z = 0; z < FOut; ++z)
		bias(z) = (rand() % 200 - 100) / 100.;

	for (int q = 0; q < FOut; ++q)
		for (int z = 0; z < FIn; ++z)
			for (int y = 0; y < K + 1; ++y)
				for (int x = 0; x < K + 1; ++x)
					filter(x, y, z, q) = (rand() % 200 - 100) / 100.;

	for (int z = 0; z < FOut; ++z)
		bias2(z) = bias(z);

	for (int q = 0; q < FOut; ++q)
		for (int z = 0; z < FOut; ++z)
			for (int y = 0; y < K + 1; ++y)
				for (int x = 0; x < K + 1; ++x)
					filter2(x, y, z, q) = (rand() % 200 - 100) / 100.;

	std::cout << "\t\tBuffers initialized" << std::endl;

	// Initialize parameters[]
	parameters(0) = N;
	parameters(1) = K;
	parameters(2) = FIn;
	parameters(3) = FOut;
	parameters(4) = BATCH_SIZE;

	for (int i = 0; i < NB_TESTS; i++)
	{
		auto start1 = std::chrono::high_resolution_clock::now();
		vgg_block(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(),
				  bias.raw_buffer(), conv.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(),
				  conv2.raw_buffer(), output.raw_buffer());
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end1 - start1;
		duration_vector_2.push_back(duration);
	}
	std::cout << "\t\tTiramisu vgg Block duration"
			  << ": " << median(duration_vector_2) << "; " << std::endl;

	std::ofstream resultfile;
	resultfile.open("tiramisu_result.txt");

	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FOut; ++z)
			for (int y = 0; y < N - 2 * K; ++y)
				for (int x = 0; x < N - 2 * K; ++x)
					resultfile << output(x, y, z, n) << "\n";

	resultfile.close();

	std::cout << "\t\t Result"
			  << ":\n\n";

	std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");
	// infile2("mkldnn_result.txt");
	std::string line1, line2;
	float file_count = 0, corr = 0, f1, f2;

	while (std::getline(infile1, line1))
	{
		std::getline(infile2, line2);
		file_count += 1;
		f1 = std::stof(line1);
		f2 = std::stof(line2);

		if (abs(f1 - f2) < 0.02)
			corr += 1;
	}

	printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

	return 0;
}
