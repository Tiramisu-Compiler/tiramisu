#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <tiramisu/utils.h>

#include "configure.h"
#include "conv_relu_fc_softmax_generator_tiramisu.o.h"

int main(int, char**)
{
	int OUTPUT_N = (N-K+2*PADDING)/STRIDE + 1;
	int FC_INPUT_SIZE = OUTPUT_N * OUTPUT_N * FOut;

	Halide::Buffer<float> input(N+2*PADDING, N+2*PADDING, FIn, BATCH_SIZE);
	Halide::Buffer<float> filter(K, K, FIn, FOut);
	Halide::Buffer<float> bias(FOut);

	Halide::Buffer<float> fc_weights(FC_INPUT_SIZE, FC_OUTPUT_SIZE);
	Halide::Buffer<float> fc_bias(FC_OUTPUT_SIZE);

	Halide::Buffer<float> result(FC_OUTPUT_SIZE, BATCH_SIZE);
	init_buffer(result, (float) 0);

	std::vector<double> duration_vector;

	for (int n=0; n < BATCH_SIZE; ++n)
		for (int z=0; z < FIn; ++z)
			for (int y=0; y < N+2*PADDING; ++y)
				for (int x=0; x < N+2*PADDING; ++x)
					if(y<PADDING || y>=N+PADDING || x<PADDING || x>=N+PADDING)
						input(x, y, z, n) = 0;
					else
						input(x, y, z, n) = 0.001;

	for (int z = 0; z < FOut; z++)
		bias(z) = 0.001;

	for (int q=0; q<FOut; q++)
		for (int z=0; z<FIn; z++)
			for (int y=0; y<K; y++)
				for (int x=0; x<K; x++)
					filter(x, y, z, q) = 0.001;

	float val=0.000001;
	for (int y=0; y < FC_OUTPUT_SIZE; y++)
		for (int x=0; x < FC_INPUT_SIZE; x++){
			fc_weights(x, y) = val;
			val+=0.0000001;
		}

	for (int z = 0; z < FC_OUTPUT_SIZE; z++)
		fc_bias(z) = 0.001;

	std::cout << "\t\tBuffers initialized" << std::endl;

	for (int i=0; i<NB_TESTS; i++)
	{
		double start = rtclock();

		conv_relu_fc_softmax(input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), fc_weights.raw_buffer(), fc_bias.raw_buffer(), result.raw_buffer());

		double end = rtclock();
		duration_vector.push_back((end - start) * 1000);
	}

	std::cout << "\t\tTiramisu Conv-Relu-FC-Softmax" << ": " << median(duration_vector) << "; " << std::endl;
	if (SHOW_OUTPUT){
		std::cout << "\t\tResult" << ": "<< std::endl;
		for(int n=0; n<BATCH_SIZE; n++){
			for(int z=0; z<FC_OUTPUT_SIZE; z++)
				std::cout << result(z, n) << ", ";
			std::cout << std::endl;
		}
	}
	if (WRITE_RESULT_TO_FILE){
		// Write results to file
		FILE* f = fopen("tiramisu_result.txt", "w");
		if (f == NULL) {
			printf("Error creating tiramisu_result.txt.\n");
			return 0;
		}

		for(int n=0; n<BATCH_SIZE; n++)
			for(int z=0; z<FC_OUTPUT_SIZE; z++)
				fprintf(f, "%.17g\n", result(z, n));

		fclose(f);
	}

	if (CHECK_CORRECTNESS){
		// Compare results with Intel MKLDNN
		std::ifstream mkldnn_result("mkldnn_result.txt");
		double tmp;
		long nb_correct = 0;

		for(int n=0; n<BATCH_SIZE; n++)
			for(int z=0; z<FC_OUTPUT_SIZE; z++){
				mkldnn_result >> tmp;
				if (std::abs(result(z, n) - tmp) <= 0.000001)
					nb_correct++;
			}

		std::cout << "\t\tResult"
			<< ":\n\n";

		std::cout << "\t\tPercentage of correctness " << 100*(((double)nb_correct)/(BATCH_SIZE * FC_OUTPUT_SIZE)) << "%" << std::endl << std::endl;
	}
	return 0;
}
