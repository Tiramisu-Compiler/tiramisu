#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "configure.h"
#include "wrapper_vgg.h"
#include <tiramisu/utils.h>
using namespace std;

int main(int, char**)
{

    Halide::Buffer<float> input(N+K, N+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K+1, K+1, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> conv(N, N, FOut, BATCH_SIZE);


    Halide::Buffer<float> filter2(K+1, K+1, FOut, FOut);
    Halide::Buffer<float> bias2(FOut);

    Halide::Buffer<float> conv2_tiramisu(N-K, N-K, FOut, BATCH_SIZE);
    Halide::Buffer<float> vgg_tiramisu_buff(N-2*K, N-2*K, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);
    Halide::Buffer<float> negative_slope(1);negative_slope(0) = 1;
    // Buffer for Halide 
    Halide::Buffer<float> vgg_halide(N-2*K, N-2*K, FOut, BATCH_SIZE);

 

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    /****************************************** Initialize Buffers *********************************************/
   srand (1);
  for (int n = 0; n < BATCH_SIZE; ++n)
	for (int z = 0; z < FIn; ++z)
		for (int y = 0; y < N+K; ++y)	       
			for (int x = 0; x < N+K; ++x)
			    //input(x, y, z, n) = rand()%10; 
				input(x, y, z, n) = 1; 

    for (int z = 0; z < FOut; ++z)
        bias(z) = 0;

     for (int y = 0; y < K+1; ++y)
        for (int x = 0; x < K+1; ++x)
            for (int z = 0; z < FIn; ++z)
		for (int q = 0; q < FOut; ++q)
		    filter(x, y, z, q) = 1;

     for (int z = 0; z < FOut; ++z)
        bias2(z) = 0;

     for (int y = 0; y < K+1; ++y)
        for (int x = 0; x < K+1; ++x)
            for (int z = 0; z < FOut; ++z)
		for (int q = 0; q < FOut; ++q)
		    filter2(x, y, z, q) = 1;

    std::cout << "\t\tBuffers initialized" << std::endl;
  


    /****************************************** Halide Part ********************************************************/

   for (int i=0; i<NB_TESTS; i++)
    {

        auto start1 = std::chrono::high_resolution_clock::now();
        vgg_ref(input.raw_buffer(),filter.raw_buffer(), bias.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(), vgg_halide.raw_buffer());
        
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_1.push_back(duration);
    }

    std::cout << "\t\tHalide vgg duration" << ": " << median(duration_vector_1)/1000 << "; " << std::endl;
  
    // Write the result 
    /*std::ofstream halide_resultfile;
    halide_resultfile.open ("/home/dina/tiramisuOut/vgg_halide_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N-2*K; ++y)	       
                for (int x = 0; x < N-2*K; ++x) halide_resultfile <<vgg_halide(x, y, z, n);  
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
        vgg_tiramisu(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(), bias.raw_buffer(), conv.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(), conv2_tiramisu.raw_buffer(),vgg_tiramisu_buff.raw_buffer(),negative_slope.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }

    std::cout << "\t\tTiramisu vgg duration" << ": " << median(duration_vector_2)/1000 << "; " << std::endl;

       // Write the result 
    std::ofstream resultfile;
    resultfile.open ("/home/dina/tiramisuOut/vgg_tiramisu_result.txt");
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z) 
            for (int y = 0; y < N-2*K; ++y)	       
                for (int x = 0; x < N-2*K; ++x) resultfile <<vgg_tiramisu_buff(x, y, z, n);     
    resultfile.close();
  

    /*************************** Comparaison of the result of  Halide & Tiramisu******************************/

     compare_4D_buffers("comparing Tiramisu output with Halide output", vgg_tiramisu_buff, vgg_halide, 5);
    return 0;
}