#include "Halide.h"
#include "wrapper_syr2.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <chrono>

#define NN 100
#define alpha 3


using namespace std;
using namespace std::chrono;

int main(int, char **)
{
    Halide::Buffer<uint8_t> A_buf(NN, NN);
    Halide::Buffer<uint8_t> x_buf(NN);
    Halide::Buffer<uint8_t> y_buf(NN);

    //output syr2
    Halide::Buffer<uint8_t> output1_buf(NN, NN);

    // Initialize matrix A with pseudorandom values:
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            A_buf(j, i) = (i + 3) * (j + 1);
        }
    }

    // Initialize Vector X with pseudorandom values:
    for (int i = 0; i < NN; i++) {
        x_buf(i) = 2;
    }

    // Initialize Vector Y with pseudorandom values:
    for (int j = 0; j < NN; j++) {
    	y_buf(j) = j;
    }

    // TRAMISU CODE EXECUTION STARTS:
    auto start1 = std::chrono::high_resolution_clock::now();

    syr2(A_buf.raw_buffer(), x_buf.raw_buffer(), y_buf.raw_buffer(), output1_buf.raw_buffer() );

    auto end1 = std::chrono::high_resolution_clock::now();
    auto  duration1 =duration_cast<microseconds>(end1 - start1);
    // TRAMISU CODE EXECUTION ENDS.

    // REFERENCE Output buffer
    Halide::Buffer<uint8_t> output2_buf(NN, NN);
    init_buffer(output2_buf, (uint8_t)0);

    // REFERENCE C++ CODE EXECUTION STARTS
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
	    output2_buf(j, i) = alpha * x_buf(i) * y_buf(j) + alpha * x_buf(j) * y_buf(i) + A_buf(j, i) ;
        }
    }
    // REFERENCE C++ CODE EXECUTION ENDS.

    auto end2 = std::chrono::high_resolution_clock::now();
    auto  duration2 =duration_cast<microseconds>(end2 - start2);

    //===== printing REFERECE EXEC TIME: =====
    std::cout << "\n REF RESOLUTION TIME : " << duration2.count() << "microseconds";
   //===== printing TIRAMISU EXEC TIME: =====
    std::cout << "\n TIRAMISU RESOLUTION TIME : " << duration1.count() << "microseconds";
    printf("\n");

   //===== Verify if TIRAMISU output is correct: =====
    compare_buffers("syr2", output1_buf, output2_buf);

    return 0;
}
