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

int main(int, char**)
{
    Halide::Buffer<float> input(N+K, N+K, FIn, BATCH_SIZE);
    Halide::Buffer<float> filter(K+1, K+1, FIn, FOut);
    Halide::Buffer<float> bias(FOut);
    Halide::Buffer<float> conv(N, N, FOut, BATCH_SIZE);
    Halide::Buffer<float> filter2(K+1, K+1, FOut, FOut);
    Halide::Buffer<float> bias2(FOut);
    Halide::Buffer<float> conv2_tiramisu(N-K, N-K, FOut, BATCH_SIZE);
    Halide::Buffer<float> output(N-2*K, N-2*K, FOut, BATCH_SIZE);
    Halide::Buffer<int> parameters(5);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;


    for (int i=0; i<NB_TESTS; i++)
    {

    	srand (1);
    	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FIn; ++z)
			for (int y = 0; y < N+K; ++y)	       
				for (int x = 0; x < N+K; ++x)
				    input(x, y, z, n) = rand()%10; 

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
            		for (int z = 0; z < FIn; ++z)
				for (int q = 0; q < FOut; ++q)
				       filter2(x, y, z, q) = 1;

    	std::cout << "\t\tBuffers initialized" << std::endl;
    


    	// Initialize parameters[]
	parameters(0) = N;
	parameters(1) = K;
	parameters(2) = FIn;
	parameters(3) = FOut;
	parameters(4) = BATCH_SIZE;


        auto start1 = std::chrono::high_resolution_clock::now();

	vgg_block(parameters.raw_buffer(), input.raw_buffer(), filter.raw_buffer(),
        bias.raw_buffer(), conv.raw_buffer(), filter2.raw_buffer(), bias2.raw_buffer(),
        conv2_tiramisu.raw_buffer(),output.raw_buffer());

    	std::ofstream resultfile;
    	resultfile.open ("tiramisu_result.txt");


    	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int z = 0; z < FOut; ++z) 
			for (int y = 0; y < N-2*K; ++y)	       
				for (int x = 0; x < N-2*K; ++x)	
    					resultfile <<output(x, y, z, n);

    	resultfile.close();
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end1 - start1;
	duration_vector_2.push_back(duration);
    }


    std::cout << "\t\tTiramisu vgg Block duration" << ": " << median(duration_vector_2) << "; " << std::endl;
    std::cout << "\t\t Result" << ": ";

    ifstream in("tiramisu_result.txt"); 
    ifstream in2("mkldnn_result.txt");

    while ((!in.eof()) && (!in2.eof())) { 
    	string line,line2;
	getline(in,line); 
	getline(in2,line2);
        if(line!=line2){ printf("error !!"); break;}
	else {	if ((in.eof()) && (in2.eof()) ) printf("\n\n correct\n\n"); }
    }
                           
    in.close(); in2.close();   

    return 0;
}
