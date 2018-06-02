#include "wrapper_laplacian.h"
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

    Halide::Buffer<uint16_t> input = Halide::Tools::load_image("./utils/images/rgb.png");

    Halide::Buffer<uint16_t> output1(input.width(), input.height(), input.channels());
    Halide::Buffer<uint16_t> output2(input.width(), input.height(), input.channels());

    // Warm up
    laplacian_tiramisu(input.raw_buffer(), output1.raw_buffer());
    laplacian_ref(input.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        laplacian_tiramisu(input.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        laplacian_ref(input.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "laplacian",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
      compare_buffers("Blurxy",  output1, output2);

    Halide::Tools::save_image(output1, "./build/laplacian_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/laplacian_ref.png");

    return 0;
}
