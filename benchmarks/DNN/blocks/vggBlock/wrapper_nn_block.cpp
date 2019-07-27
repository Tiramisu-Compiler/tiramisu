#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <iomanip>
#include "configure.h"
#include "generated_vgg_block.o.h"
#include <tiramisu/utils.h>
using namespace std;

int main(int, char **)
{
	Halide::Buffer<float> input(FIN1_BLOCKING, N, N, FIN1_NB_BLOCKS, BATCH_SIZE);

	Halide::Buffer<float> filter1(FIN2_BLOCKING, FIN1_BLOCKING, K, K, FIN1_NB_BLOCKS, FIN2_NB_BLOCKS);
	Halide::Buffer<float> bias1(FOut);

	Halide::Buffer<float> filter2(FOUT_BLOCKING, FIN2_BLOCKING, K, K, FIN2_NB_BLOCKS, FOUT_NB_BLOCKS);
	Halide::Buffer<float> bias2(FOut);

	Halide::Buffer<float> output(FOUT_BLOCKING, N/2, N/2, FOUT_NB_BLOCKS, BATCH_SIZE);

	std::vector<double> duration_vector;

	// Initialize buffers
	srand(1);
	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FIn; ++z)
			for (int y = 0; y < N; ++y)
				for (int x = 0; x < N; ++x)
					input(z%FIN1_BLOCKING, x, y, z/FIN1_BLOCKING, n) = (std::rand() % 200 - 100) / 100.;

	for (int z = 0; z < FOut; ++z)
		bias1(z) = (rand() % 200 - 100) / 100.;

	for (int q = 0; q < FOut; ++q)
		for (int z = 0; z < FIn; ++z)
			for (int y = 0; y < K; ++y)
				for (int x = 0; x < K; ++x)
					filter1(q%FIN2_BLOCKING, z%FIN1_BLOCKING, x, y, z/FIN1_BLOCKING, q/FIN2_BLOCKING) = (rand() % 200 - 100) / 100.;

	for (int z = 0; z < FOut; ++z)
		bias2(z) = (rand() % 200 - 100) / 100.;

	for (int q = 0; q < FOut; ++q)
		for (int z = 0; z < FOut; ++z)
			for (int y = 0; y < K; ++y)
				for (int x = 0; x < K; ++x)
					filter2(q%FOUT_BLOCKING, z%FIN2_BLOCKING, x, y, z/FIN2_BLOCKING, q/FOUT_BLOCKING) = (rand() % 200 - 100) / 100.;

	std::cout << "\t\tBuffers initialized" << std::endl;

	// Execute Tiramisu code
	for (int i = 0; i < NB_TESTS; i++) {
		double start = rtclock();

		vgg_block(
			input.raw_buffer(), 
			filter1.raw_buffer(),
			bias1.raw_buffer(),
			filter2.raw_buffer(),
			bias2.raw_buffer(),
			output.raw_buffer()
		);

		double end = rtclock();
		duration_vector.push_back((end - start) * 1000);
	}

	std::cout << "\t\tTiramisu vgg Block duration"
			  << ": " << median(duration_vector) << "; " << std::endl;

	// Write results to file
	std::ofstream resultfile;
	resultfile.open("tiramisu_result.txt");

	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FOut; ++z)
			for (int y = 0; y < N/2; ++y)
				for (int x = 0; x < N/2; ++x)
					resultfile << setprecision(10) << output(z%FOUT_BLOCKING, x, y, z/FOUT_BLOCKING, n) << std::endl;

	resultfile.close();

	// Compare results with Intel MKL
	std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");

	std::string line1, line2;
	float file_count = 0, corr = 0, f1, f2;

	while (std::getline(infile1, line1))
	{
		std::getline(infile2, line2);
		file_count += 1;
		f1 = std::stof(line1);
		f2 = std::stof(line2);

		if (abs(f1 - f2) < 0.001)
			corr += 1;
	}

    std::cout << "\t\tResult"
              << ":\n\n";

    cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << endl << endl;

	return 0;
}
