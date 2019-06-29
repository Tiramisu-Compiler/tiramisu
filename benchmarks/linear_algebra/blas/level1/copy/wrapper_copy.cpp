#include "Halide.h"
#include "wrapper_copy.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <chrono>


#define NN 10 


using namespace std;
using namespace std::chrono;


int main(int, char **)
{
  
    // Declare vector input and initialize it with 3 
    Halide::Buffer<uint8_t> input(NN);
    init_buffer(input, (uint8_t)3);

    Halide::Buffer<uint8_t> output(NN);

    // TRAMISU CODE EXECUTION STARTS:
    auto start1 = std::chrono::high_resolution_clock::now();

    copy(input.raw_buffer(), output.raw_buffer());

    auto end1 = std::chrono::high_resolution_clock::now();
    auto  duration1 =duration_cast<microseconds>(end1 - start1);
    // TRAMISU CODE EXECUTION ENDS.

    // REFERENCE Output buffer
    Halide::Buffer<uint8_t> expected(NN);

    // REFERENCE C++ CODE EXECUTION STARTS
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NN; i++) {
            expected(i) = input(i);
        
    }
    // REFERENCE C++ CODE EXECUTION ENDS.

    auto end2 = std::chrono::high_resolution_clock::now();
    auto  duration2 =duration_cast<microseconds>(end2 - start2);

    //===== printing REFERECE EXEC TIME: =====
    std::cout << "\n REF RESOLUTION TIME : " << duration2.count() << "microseconds";
    //===== printing TIRAMISU EXEC TIME: =====
    std::cout << "\n TIRAMISU RESOLUTION TIME : " << duration1.count() << "microseconds";
    printf("\n");

    compare_buffers("copy", output, expected);

    return 0;
}
