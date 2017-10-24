#include "wrapper_heat2d.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> input(Halide::Float(32), 10000, 10000);
    // Init randomly
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            input(x, y) = random();
            input(x, y) = random();
        }
    }

    Halide::Buffer<float> output1(input.width(), input.height());
    Halide::Buffer<float> output2(input.width(), input.height());

    // Warm up code.
    heat2d_tiramisu(input.raw_buffer(), output1.raw_buffer());
    heat2d_ref(input.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        heat2d_tiramisu(input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        heat2d_ref(input.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "heat2d",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
	compare_buffers("benchmark_heat2d", output1, output2);

    return 0;
}
