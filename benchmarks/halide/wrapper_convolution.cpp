#include "wrapper_convolution.h"
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

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./utils/images/rgb.png");

    Halide::Buffer<float> kernel(3, 3);
    kernel(0,0) = 0; kernel(0,1) = 1.0f/5; kernel(0,2) = 0;
    kernel(1,0) = 1.0f/5; kernel(1,1) = 1.0f/5; kernel(1,2) = 1.0f/5;
    kernel(2,0) = 0; kernel(2,1) = 1.0f/5; kernel(2,2) = 0;

    Halide::Buffer<uint8_t> output1(input.width()-8, input.height()-8, input.channels());
    Halide::Buffer<uint8_t> output2(input.width()-8, input.height()-8, input.channels());

    // Warm up
    convolution_tiramisu(input.raw_buffer(), kernel.raw_buffer(), output1.raw_buffer());
    convolution_ref(input.raw_buffer(), kernel.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        convolution_tiramisu(input.raw_buffer(), kernel.raw_buffer(),
			output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        convolution_ref(input.raw_buffer(), kernel.raw_buffer(),
			output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "convolution",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    Halide::Tools::save_image(output1, "./build/convolution_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/convolution_ref.png");

    if (CHECK_CORRECTNESS)
        compare_buffers_approximately("convolution",  output1, output2, 1);

    return 0;
}
