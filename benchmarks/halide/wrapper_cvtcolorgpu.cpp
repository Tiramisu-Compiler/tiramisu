#include "wrapper_cvtcolorgpu.h"
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

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Buffer<int32_t> SIZES_b(2);
    SIZES_b(0) = input.extent(0);
    SIZES_b(1) = input.extent(1);
    Halide::Buffer<uint8_t> output1(input.width(), input.height());
    Halide::Buffer<uint8_t> output2(input.width(), input.height());

    // Warm up code.
    cvtcolorgpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
    cvtcolorgpu_ref(input.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        cvtcolorgpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        cvtcolorgpu_ref(input.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "cvtcolorgpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
        compare_buffers("benchmark_cvtcolorgpu", output1, output2);

    Halide::Tools::save_image(output1, "./build/cvtcolorgpu_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/cvtcolorgpu_ref.png");

    return 0;
}
