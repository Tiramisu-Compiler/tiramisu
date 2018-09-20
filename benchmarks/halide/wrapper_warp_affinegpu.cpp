#include "wrapper_warp_affinegpu.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

float random(float a, float b)
{
    return ((b - a)*((float)rand()/RAND_MAX)) + a;
}

std::chrono::duration<double, std::milli> run_halide(Halide::Buffer<uint8_t> &in, Halide::Buffer<float> &out)
{
    in(0, 0, 0) = in(0, 0, 0);
    auto start = std::chrono::high_resolution_clock::now();
    warp_affinegpu_ref(in.raw_buffer(), out.raw_buffer());
    halide_copy_to_host(nullptr, out.raw_buffer());
    halide_device_free(nullptr, out.raw_buffer());
    halide_device_free(nullptr, in.raw_buffer());
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    // TODO: Convert this image to grayscale. Halide implementation copies the whole image
    //       to GPU instead of the used region, which is three times what's needed.
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./utils/images/rgb.png");
    Halide::Buffer<int32_t> SIZES_b(2);
    SIZES_b(0) = input.extent(0);
    SIZES_b(1) = input.extent(1);

    Halide::Buffer<float> output1(input.width(), input.height());
    Halide::Buffer<float> output2(input.width(), input.height());

    // Warm up
    warp_affinegpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
    run_halide(input, output2);

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        warp_affinegpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
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
        duration_vector_2.push_back(run_halide(input, output2));
    }
    for (auto &d: duration_vector_2)
        std::cout << "," << d.count();
    std::cout << std::endl;

    print_time("performance_CPU.csv", "warp_affinegpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
        compare_buffers_approximately("benchmark_warp_affinegpu", output1, output2);

    return 0;
}
