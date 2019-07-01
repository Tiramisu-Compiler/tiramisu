#include "Halide.h"
#include "wrapper_swap.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define NN 100
int main(int, char **)
{
   // Declare vector 1 and initialize it with 3
    Halide::Buffer<uint8_t> input1_buf(NN);
    init_buffer(input1_buf,(uint8_t)3);
   // Declare vector 2 and initialize it with 4
    Halide::Buffer<uint8_t> input2_buf(NN);
    init_buffer(input2_buf,(uint8_t)4);
   // REF C++ PROGRAM STARTS:
    auto start1 = std::chrono::high_resolution_clock::now();
   // expected1 = what we expect input1 to be after calling swap
    Halide::Buffer<uint8_t> expected1(NN);
    for (int i = 0; i < NN;i++) {
            expected1(i) = input2_buf(i);
    }
   // expected2 = what we expect input2 to be after calling swap
    Halide::Buffer<uint8_t> expected2(NN);
    for (int i = 0; i < NN; i++) {
            expected2(i) = input1_buf(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto  duration1 =duration_cast<microseconds>(end1 - start1);
   // REF C++ PROGRAM ENDS.
   // TIRAMISU PROGRAM STARTS:
    auto start2 = std::chrono::high_resolution_clock::now();
    swap(input1_buf.raw_buffer(), input2_buf.raw_buffer());
    auto end2 = std::chrono::high_resolution_clock::now();
    auto  duration2 =duration_cast<microseconds>(end2 - start2);
   // TIRAMISU PROGRAM ENDS.
   //===== printing REFERECE EXEC TIME: =====
    std::cout << "\n REF RESOLUTION TIME : " << duration1.count() << "microseconds";
   //===== printing TIRAMISU EXEC TIME: =====
    std::cout << "\n TIRAMISU RESOLUTION TIME : " << duration2.count() << "microseconds";
    printf("\n");
    compare_buffers("swap_input1", input1_buf, expected1);
    printf("\n");
    compare_buffers("swap_input2", input2_buf, expected2);
    printf("\n");
    return 0;
}
