#include "wrapper_rgbyuv420gpu.h"
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
    Halide::Buffer<int32_t> size(2);
    size(0) = input.extent(0);
    size(1) = input.extent(1);

    Halide::Buffer<uint8_t> output_ref_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_ref_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_ref_v(input.width()/2, input.height()/2);

    Halide::Buffer<uint8_t> output_tiramisu_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_tiramisu_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_tiramisu_v(input.width()/2, input.height()/2);

    std::cout << "STARTING TEST\n";
    std::cout << "y size (width, height): " << output_tiramisu_y.width() << ", " << output_tiramisu_y.height() << "\n";
    std::cout << "u size (width, height): " << output_tiramisu_u.width() << ", " << output_tiramisu_u.height() << "\n";
    std::cout << "v size (width, height): " << output_tiramisu_v.width() << ", " << output_tiramisu_v.height() << "\n";
    // Warm up
    rgbyuv420gpu_ref(input.raw_buffer(), output_ref_y.raw_buffer(), output_ref_u.raw_buffer(), output_ref_v.raw_buffer());
    rgbyuv420gpu_tiramisu(size, input.raw_buffer(), output_tiramisu_y.raw_buffer(), output_tiramisu_u.raw_buffer(), output_tiramisu_v.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        rgbyuv420gpu_tiramisu(size.raw_buffer(), input.raw_buffer(), output_tiramisu_y.raw_buffer(), output_tiramisu_u.raw_buffer(), output_tiramisu_v.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rgbyuv420gpu_ref(input.raw_buffer(), output_ref_y.raw_buffer(), output_ref_u.raw_buffer(), output_ref_v.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "rgbyuv420gpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    Halide::Tools::save_image(output_tiramisu_y, "./build/rgbyuv420gpu_y_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_u, "./build/rgbyuv420gpu_u_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_v, "./build/rgbyuv420gpu_v_tiramisu.png");
    Halide::Tools::save_image(output_ref_y, "./build/rgbyuv420gpu_y_ref.png");
    Halide::Tools::save_image(output_ref_u, "./build/rgbyuv420gpu_u_ref.png");
    Halide::Tools::save_image(output_ref_v, "./build/rgbyuv420gpu_v_ref.png");

    if (CHECK_CORRECTNESS)
    {
        std::cout << "Compare y buffer\n";
    	compare_buffers("benchmark_rgbyuv420gpu", output_tiramisu_y, output_ref_y);
        std::cout << "Compare u buffer\n";
    	compare_buffers("benchmark_rgbyuv420gpu", output_tiramisu_u, output_ref_u);
        std::cout << "Compare v buffer\n";
    	compare_buffers("benchmark_rgbyuv420gpu", output_tiramisu_y, output_ref_y);
    }

    return 0;
}
