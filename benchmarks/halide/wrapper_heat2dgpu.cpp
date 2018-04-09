#include "wrapper_heat2dgpu.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

std::chrono::duration<double, std::milli> run_halide(Halide::Buffer<float> &in, Halide::Buffer<float> &out)
{
    in(0, 0) = in(0, 0);
    auto start = std::chrono::high_resolution_clock::now();
    heat2dgpu_ref(in.raw_buffer(), out.raw_buffer());
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

    Halide::Buffer<float> input(Halide::Float(32), 10000, 10000);

    Halide::Buffer<int32_t> size(2);
    size(0) = input.extent(0);
    size(1) = input.extent(1);
    // Init randomly
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            input(x, y) = 1;
            input(x, y) = 1;
        }
    }

    Halide::Buffer<float> output1(input.width(), input.height());
    Halide::Buffer<float> output2(input.width(), input.height());

    // Warm up code.
    heat2dgpu_tiramisu(size.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
    run_halide(input, output2);

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        heat2dgpu_tiramisu(size.raw_buffer(), input.raw_buffer(), output1.raw_buffer());
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

    print_time("performance_CPU.csv", "heat2dgpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
	compare_buffers_approximately("benchmark_heat2dgpu", output1, output2);

    return 0;
}
