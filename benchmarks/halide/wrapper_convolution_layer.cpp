#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include "configure.h"
#include "wrapper_convolution_layer.h"
#include <tiramisu/utils.h>


int main(int, char**)
{

    Halide::Buffer<float> input(N+K, N+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K+1, K+1, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> convolution_layer_halide(N, N, FOut, BATCH_SIZE);
    Halide::Buffer<float> convolution_layer_tiramisu_buffer(N, N, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

 /****************************************** Initialize Buffers *********************************************/
    for (int y = 0; y < N+K; ++y)
        for (int x = 0; x < N+K; ++x)
            for (int z = 0; z < FIn; ++z)
	        for (int n = 0; n < BATCH_SIZE; ++n)
		      input(x, y, z, n) = 1;

    for (int z = 0; z < FIn; ++z)
        bias(z) = 1;

     for (int y = 0; y < K+1; ++y)
        for (int x = 0; x < K+1; ++x)
            for (int z = 0; z < FIn; ++z)
		for (int q = 0; q < FOut; ++q)
		    filter(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;

  /****************************************** Halide Part ********************************************************/

    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        convolution_layer_ref(input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), convolution_layer_halide.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_1.push_back(duration);
    }

    std::cout << "\t\tHalide convolution_layer" << ": " << median(duration_vector_1) << "; " << std::endl;

      // Write the result 
    std::ofstream hdresuts;
    hdresuts.open ("/home/dina/tiramisuOut/convolution_layer_halide_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N; ++y)	       
                for (int x = 0; x < N; ++x) hdresuts <<convolution_layer_halide(x, y, z, n);     
    hdresuts.close();

    /****************************************** Tiramisu Part ********************************************************/
    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = K;
    parameters(2) = FIn;
    parameters(3) = FOut;
    parameters(4) = BATCH_SIZE;

    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();

        convolution_layer_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), convolution_layer_tiramisu_buffer.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }

    std::cout << "\t\tTiramisu convolution_layer" << ": " << median(duration_vector_2) << "; " << std::endl;
  
     // Write the result 
    std::ofstream resultfile;
    resultfile.open ("/home/dina/tiramisuOut/convolution_layer_tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N ;++y)	       
                for (int x = 0; x < N; ++x) resultfile <<convolution_layer_tiramisu_buffer(x, y, z, n);     
    resultfile.close();


     /*************************** Comparaison of the result of  Halide & Tiramisu**********************************************/
    compare_buffers("comparing Tiramisu output with Halide output", convolution_layer_tiramisu_buffer, convolution_layer_halide);


    return 0;
}