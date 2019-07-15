#include "wrapper_resize.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./utils/images/gray.png");

    Halide::Buffer<float> output1(input.width()/1.5f, input.height()/1.5f);
    Halide::Buffer<float> output2(input.width()/1.5f, input.height()/1.5f);

    Halide::Buffer<uint8_t> output1_int(input.width()/1.5f, input.height()/1.5f);
    Halide::Buffer<uint8_t> output2_int(input.width()/1.5f, input.height()/1.5f);

    // Tiramisu
    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();

        resize_tiramisu(input.raw_buffer(), output1.raw_buffer());

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();

        resize_ref(input.raw_buffer(), output2.raw_buffer());

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "resize",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    // Convert float buffers to int buffers in order to save them as images using Halide
    for (int y = 0; y < input.height()/1.5f; ++y)
        for (int x = 0; x < input.width()/1.5f; ++x)
            output1_int(x, y) = output1(x, y);

    Halide::Tools::save_image(output1_int, "./build/resize_tiramisu.png");

    for (int y = 0; y < input.height()/1.5f; ++y)
        for (int x = 0; x < input.width()/1.5f; ++x)
            output2_int(x, y) = output2(x, y);

    Halide::Tools::save_image(output2_int, "./build/resize_ref.png");

    return 0;
}
