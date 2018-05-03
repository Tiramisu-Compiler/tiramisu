#include "wrapper_convolutiongpu.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

std::chrono::duration<double, std::milli> run_halide(Halide::Buffer<uint8_t> &in, Halide::Buffer<float> &kernel, Halide::Buffer<uint8_t> &out)
{
    in(0, 0, 0) = in(0, 0, 0);
    kernel(0, 0) = kernel(0, 0);
    auto start = std::chrono::high_resolution_clock::now();
    convolutiongpu_ref(in.raw_buffer(), kernel.raw_buffer(), out.raw_buffer());
    halide_copy_to_host(nullptr, out.raw_buffer());
    halide_device_free(nullptr, out.raw_buffer());
    halide_device_free(nullptr, in.raw_buffer());
    halide_device_free(nullptr, kernel.raw_buffer());
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");
    Halide::Buffer<int32_t> sizes(2);
    sizes(0) = input.extent(0);
    sizes(1) = input.extent(1);

    Halide::Buffer<float> kernel(3, 3);
    kernel(0,0) = 0; kernel(0,1) = 1.0f/5; kernel(0,2) = 0;
    kernel(1,0) = 1.0f/5; kernel(1,1) = 1.0f/5; kernel(1,2) = 1.0f/5;
    kernel(2,0) = 0; kernel(2,1) = 1; kernel(2,2) = 0;

    Halide::Buffer<uint8_t> output1(input.width()-2, input.height()-2, input.channels());
    Halide::Buffer<uint8_t> output2(input.width()-2, input.height()-2, input.channels());

    // Warm up
    convolutiongpu_tiramisu(sizes.raw_buffer(), input.raw_buffer(), kernel.raw_buffer(), output1.raw_buffer());
    run_halide(input, kernel, output2);

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        convolutiongpu_tiramisu(sizes.raw_buffer(), input.raw_buffer(), kernel.raw_buffer(),
			output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    for (auto &d: duration_vector_1)
        std::cout << "," << d.count();
    std::cout << std::endl;

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        duration_vector_2.push_back(run_halide(input, kernel, output2));
    }

    for (auto &d: duration_vector_2)
        std::cout << "," << d.count();
    std::cout << std::endl;

    print_time("performance_CPU.csv", "convolutiongpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    Halide::Tools::save_image(output1, "./build/convolutiongpu_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/convolutiongpu_ref.png");

    if (CHECK_CORRECTNESS)
        compare_buffers("convolutiongpu",  output1, output2);

    return 0;
}
