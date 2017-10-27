#include "wrapper_rgbyuv420.h"
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

    Halide::Buffer<uint8_t> output_ref_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_ref_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_ref_v(input.width()/2, input.height()/2);

    Halide::Buffer<uint8_t> output_tiramisu_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_tiramisu_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_tiramisu_v(input.width()/2, input.height()/2);

    // Warm up
    rgbyuv420_tiramisu(input.raw_buffer(), output_tiramisu_y.raw_buffer(), output_tiramisu_u.raw_buffer(), output_tiramisu_v.raw_buffer());
    rgbyuv420_ref(input.raw_buffer(), output_ref_y.raw_buffer(), output_ref_u.raw_buffer(), output_ref_v.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        rgbyuv420_tiramisu(input.raw_buffer(), output_tiramisu_y.raw_buffer(), output_tiramisu_u.raw_buffer(), output_tiramisu_v.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rgbyuv420_ref(input.raw_buffer(), output_ref_y.raw_buffer(), output_ref_u.raw_buffer(), output_ref_v.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "rgbyuv420",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
    {
	compare_buffers("benchmark_rgbyuv420", output_tiramisu_y, output_ref_y);
	compare_buffers("benchmark_rgbyuv420", output_tiramisu_u, output_ref_u);
	compare_buffers("benchmark_rgbyuv420", output_tiramisu_y, output_ref_y);
    }

    Halide::Tools::save_image(output_tiramisu_y, "./build/rgbyuv420_y_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_u, "./build/rgbyuv420_u_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_v, "./build/rgbyuv420_v_tiramisu.png");
    Halide::Tools::save_image(output_ref_y, "./build/rgbyuv420_y_ref.png");
    Halide::Tools::save_image(output_ref_u, "./build/rgbyuv420_u_ref.png");
    Halide::Tools::save_image(output_ref_v, "./build/rgbyuv420_v_ref.png");

    return 0;
}
