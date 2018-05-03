#include "wrapper_gaussiangpu.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

std::chrono::duration<double, std::milli> run_halide(Halide::Buffer<uint8_t> &in, Halide::Buffer<float> &kernelX, Halide::Buffer<float> &kernelY, Halide::Buffer<uint8_t> &out)
{
    in(0, 0, 0) = in(0, 0, 0);
    kernelX(0) = kernelX(0);
    kernelY(0) = kernelY(0);
    auto start = std::chrono::high_resolution_clock::now();
    gaussiangpu_ref(in.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), out.raw_buffer());
    halide_copy_to_host(nullptr, out.raw_buffer());
    halide_device_free(nullptr, out.raw_buffer());
    halide_device_free(nullptr, in.raw_buffer());
    halide_device_free(nullptr, kernelX.raw_buffer());
    halide_device_free(nullptr, kernelY.raw_buffer());
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");
    Halide::Buffer<int32_t> SIZES_b(3);
    SIZES_b(0) = input.extent(0);
    SIZES_b(1) = input.extent(1);
    SIZES_b(2) = input.extent(2);

    Halide::Buffer<float> kernelX(5);
    Halide::Buffer<float> kernelY(5);

    kernelX(0) = 1.0f; kernelX(1) = 4.0f; kernelX(2) = 6.0f; kernelX(3) = 4.0f; kernelX(4) = 1.0f;
    kernelY(0) = 1.0f/256; kernelY(1) = 4.0f/256; kernelY(2) = 6.0f/256; kernelY(3) = 4.0f/256; kernelY(4) = 1.0f/256;

    Halide::Buffer<uint8_t> output1(input.width()-5, input.height()-5, input.channels());
    Halide::Buffer<uint8_t> output2(input.width()-5, input.height()-5, input.channels());

    //Warm up
    gaussiangpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), output1.raw_buffer());
    run_halide(input, kernelX, kernelY, output2);

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        gaussiangpu_tiramisu(SIZES_b.raw_buffer(), input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), output1.raw_buffer());
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
        duration_vector_2.push_back(run_halide(input, kernelX, kernelY, output2));
    }

    for (auto &d: duration_vector_2)
        std::cout << "," << d.count();
    std::cout << std::endl;


    print_time("performance_CPU.csv", "gaussiangpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
	compare_buffers("Gaussian",  output1, output2);

    Halide::Tools::save_image(output1, "./build/gaussiangpu_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/gaussiangpu_ref.png");

    return 0;
}
