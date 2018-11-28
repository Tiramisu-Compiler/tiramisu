#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "wrapper_convolution_layer.h"
#include <tiramisu/utils.h>
using namespace std;

int main(int, char**)
{

    Halide::Buffer<float> input(N+K, N+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K+1, K+1, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> convolution_layer_halide(N, N, FOut, BATCH_SIZE);
    Halide::Buffer<float> convolution_layer_tiramisu_buff(N, N, FOut, BATCH_SIZE);

    Halide::Buffer<int> parameters(5);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    /****************************************** Initialize Buffers *********************************************/
   srand (1);
  for (int n = 0; n < BATCH_SIZE; ++n)
	for (int z = 0; z < FIn; ++z)
		for (int y = 0; y < N+K; ++y)	       
			for (int x = 0; x < N+K; ++x)
			    //input(x, y, z, n) = rand()%10; 
				input(x, y, z, n) = 5; 

    for (int z = 0; z < FOut; ++z)
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
        convolution_layer_ref(input.raw_buffer(),filter.raw_buffer(), bias.raw_buffer(),convolution_layer_halide.raw_buffer());
        
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_1.push_back(duration);
    }

    std::cout << "\t\tHalide convolution_layer duration" << ": " << median(duration_vector_1)/1000 << "; " << std::endl;
  
    // Write the result 
   /* std::ofstream halide_resultfile;
    halide_resultfile.open ("/home/dina/tiramisuOut/convolution_layer_halide_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N; ++y)	       
                for (int x = 0; x < N; ++x) halide_resultfile <<convolution_layer_halide(x, y, z, n);  
    halide_resultfile.close();*/

    /****************************************** Tiramisu Part ********************************************************/

    // Initialize parameters[]
    parameters(0) = N;
    parameters(1) = K;
    parameters(2) = FIn;
    parameters(3) = FOut;
    parameters(4) = BATCH_SIZE;


    for (int i=0; i<NB_TESTS; i++)
    {
       // srand (1);
        auto start1 = std::chrono::high_resolution_clock::now();
        convolution_layer_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(),convolution_layer_tiramisu_buff.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }

    std::cout << "\t\tTiramisu convolution_layer duration" << ": " << median(duration_vector_2)/1000 << "; " << std::endl;

       // Write the result 
    /*std::ofstream resultfile;
    resultfile.open ("/home/dina/tiramisuOut/convolution_layer_tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N; ++y)	       
                for (int x = 0; x < N; ++x) resultfile <<convolution_layer_tiramisu_buff(x, y, z, n);     
    resultfile.close();*/
  

    /*************************** Comparaison of the result of  Halide & Tiramisu******************************/

     compare_4D_buffers("comparing Tiramisu output with Halide output", convolution_layer_tiramisu_buff, convolution_layer_halide, 5);
    return 0;
}