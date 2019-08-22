#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "generated_conv_layer.o.h"
#include <tiramisu/utils.h>

int main(int, char**)
{
	srand(1);
	std::vector<double> duration_vector;
	double start, end;

	Halide::Buffer<float> input(FIN_BLOCKING, N + 2, N + 2, FIN_NB_BLOCKS, BATCH_SIZE);
	Halide::Buffer<float> filter(FOUT_BLOCKING, FIN_BLOCKING, K, K, FIN_NB_BLOCKS, FOUT_NB_BLOCKS);
	Halide::Buffer<float> filter2(FOUT_BLOCKING, FIN_BLOCKING, 3, FIN_NB_BLOCKS, FOUT_NB_BLOCKS);
	Halide::Buffer<int8_t> zero_weight_filters_per_output_channel(FOut);
	Halide::Buffer<float> bias(FOut);
	Halide::Buffer<float> conv(FOUT_BLOCKING, N, N, FOUT_NB_BLOCKS, BATCH_SIZE);


	// Initialize buffers
	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int fin = 0; fin < FIn; ++fin)
			for (int y = 0; y < N + 2; ++y)
				for (int x = 0; x < N + 2; ++x)
					input(fin%FIN_BLOCKING, x, y, fin/FIN_BLOCKING, n) = ((float)(rand()%256 - 128)) / 127.f;

    	for (int fout = 0; fout < FOut; ++fout)
        {
		int zero_weights = 0;
		for (int fin = 0; fin < FIn; ++fin)
		{
			for (int k_y = 0; k_y < K; ++k_y)
				for (int k_x = 0; k_x < K; ++k_x)
				{
					if (zero_weights < zero_weight_filters_per_output_channel[fout])
	 				        filter(fout%FOUT_BLOCKING, fin%FIN_BLOCKING, k_x, k_y, fin/FIN_BLOCKING, fout/FOUT_BLOCKING) = 0;
					else
						filter(fout%FOUT_BLOCKING, fin%FIN_BLOCKING, k_x, k_y, fin/FIN_BLOCKING, fout/FOUT_BLOCKING) = ((float)(rand()%256 - 128)) / 127.f;
				}

			zero_weights++;
		}
    	}

	for (int fout = 0; fout < FOut; ++fout)
		bias(fout) = ((float)(rand()%256 - 128)) / 127.f;

	std::cout << "\t\tBuffers initialized" << std::endl;

	// Execute Tiramisu code
	for (int i = 0; i < NB_TESTS; i++) {
		start = rtclock();

		conv_tiramisu(
			input.raw_buffer(), 
			filter.raw_buffer(), 
			filter2.raw_buffer(),
			zero_weight_filters_per_output_channel.raw_buffer(),
			bias.raw_buffer(), 
			conv.raw_buffer()
		);
			
		end = rtclock();
		duration_vector.push_back((end - start) * 1000);
	}

	std::cout << "\t\tN = " << N << "; BATCH_SIZE = " << BATCH_SIZE << "; FIn = " << FIn << "; FOut = " << FOut << ";" << std::endl;
	std::cout << "\t\tTiramisu conv" << ": " << median(duration_vector) << "; " << std::endl;

	// Write results to file
	FILE* f = fopen("tiramisu_result.txt", "w");
	if (f == NULL) {
		printf("Error creating tiramisu_result.txt.\n");
		return 0;
	}

	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int fout = 0; fout < FOut; ++fout)
			for (int y = 0; y < N; ++y)
				for (int x = 0; x < N; ++x)
					fprintf(f, "%.10g\n", conv(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n));

	fclose(f);

    // Compare results with Intel MKL
    std::ifstream mkl_result("mkl_result.txt");
    float tmp;
    float file_count = 0, corr = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    mkl_result >> tmp;

                    file_count++;
                    if (std::abs(conv(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) <= 0.0001)
                        corr++;
                }

    std::cout << "\t\tResult"
              << ":\n\n";

    std::cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << std::endl << std::endl;

    return 0;
}
