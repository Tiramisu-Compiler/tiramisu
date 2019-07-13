#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <tiramisu/utils.h>

#include "configure.h"
#include "transpose_conv_generator_tiramisu.o.h"

int main(int, char**)
{
    int OUTPUT_N = (N-1)*STRIDE + K;
    Halide::Buffer<float> input(N, N, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K, K, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> result(OUTPUT_N, OUTPUT_N, FOut, BATCH_SIZE);
	init_buffer(result, (float) 0);
	
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;
	
    for (int n=0; n < BATCH_SIZE; n++)
		for (int z=0; z < FIn; z++)
			for (int y=0; y < N; y++)
				for (int x=0; x < N; x++)
					input(x, y, z, n) = 1;
	
    for (int z = 0; z < FOut; z++)
	bias(z) = 1;
	
    for (int q=0; q < FOut; q++)
		for (int z=0; z < FIn; z++)
			for (int y=0; y < K; y++)
				for (int x=0; x < K; x++)
					filter(x, y, z, q) = 1;
	
    std::cout << "\t\tBuffers initialized" << std::endl;
	
    for (int i=0; i < NB_TESTS; i++)
    {
		auto start1 = std::chrono::high_resolution_clock::now();
		transpose_conv(input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), result.raw_buffer());
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double,std::milli> duration = end1 - start1;
		duration_vector_2.push_back(duration);
	}
	
    std::cout << "\t\tTiramisu Deconv time : " << median(duration_vector_2) << "; " << std::endl;
	
	// Print the output
    if(SHOW_OUTPUT){
		std::cout << "\t\tResult" << ": "<< std::endl;
  		for(int n=0; n < BATCH_SIZE; n++)
		{
  			for(int z=0; z < FOut; z++)
			{
  				for(int y=PADDING; y < OUTPUT_N-PADDING; y++)
				{
					for(int x=PADDING; x < OUTPUT_N-PADDING; x++)
						std::cout << result(x, y, z, n) << ", ";
  					std::cout << std::endl;
				}
  				std::cout << std::endl;
			}
  			std::cout << std::endl;
		}
		
		
  		std::cout << std::endl << "\t\tInput" << ": "<< std::endl;
		
  		for(int n=0; n < BATCH_SIZE; n++)
		{
  			for(int z=0; z < FIn; z++)
			{
  				for(int y=0 ; y < N; y++)
				{
  					for(int x=0; x < N; x++)
						std::cout << input(x, y, z, n) << ", ";
  					std::cout << std::endl;
				}
  				std::cout << std::endl;
			}
  			std::cout << std::endl;
		}
	}
	
	// Print result to a file and compare results with MKLDNN
    if(SAVE_TO_FILE_AND_COMPARE){
		// Write results to file
		FILE* f = fopen("tiramisu_result.txt", "w");
		if (f == NULL) {
			printf("Error creating tiramisu_result.txt.\n");
			return 0;
		}
		
		for(int n=0; n < BATCH_SIZE; n++)
			for(int z=0; z < FOut; z++)
				for(int y=PADDING; y < OUTPUT_N - PADDING; y++)
					for(int x=PADDING; x < OUTPUT_N - PADDING; x++)
						fprintf(f, "%.17g\n", result(x, y, z, n));
		
		fclose(f);
		
		// Compare results with Intel MKLDNN
		std::ifstream mkl_result("mkldnn_result.txt");
		double tmp;
		long nb_correct = 0;
		
		for(int n=0; n < BATCH_SIZE; n++)
			for(int z=0; z < FOut; z++)
				for(int y=PADDING; y < OUTPUT_N - PADDING; y++)
					for(int x=PADDING; x < OUTPUT_N - PADDING; x++)
					{
						mkl_result >> tmp;
						if (abs(result(x, y, z, n) - tmp) <= 0.00000001)
							nb_correct++;
					}
		
		std::cout << "\t\tResult"
		<< ":\n\n";
		
		std::cout << "\t\tPercentage of correctness " << 100*(((double)nb_correct)/(BATCH_SIZE*FOut*(OUTPUT_N-2*PADDING)*(OUTPUT_N-2*PADDING))) << "%" << std::endl << std::endl;
		
	}
	
	
	
    return 0;
}
