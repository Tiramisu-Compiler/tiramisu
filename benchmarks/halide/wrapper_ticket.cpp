#include "wrapper_ticket.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

#include <tiramisu/utils.h>

#define N 1024
#define NB_TESTS 60
#define CHECK_CORRECTNESS 1

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> output1(N);
    Halide::Buffer<int> output2(N);

    init_buffer(output1, (uint8_t) 0);
    init_buffer(output2, (int) 0);

    //Warm up
    ticket_tiramisu(output1.raw_buffer());
    ticket_ref(output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        ticket_tiramisu(output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        ticket_ref(output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "edge",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    return 0;
}
