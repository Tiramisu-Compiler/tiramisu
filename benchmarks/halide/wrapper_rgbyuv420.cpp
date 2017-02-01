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

    Halide::Buffer<int16_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Buffer<uint8_t> output_ref_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_ref_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_ref_v(input.width()/2, input.height()/2);

    Halide::Buffer<uint8_t> output_tiramisu_y(input.width(), input.height());
    Halide::Buffer<uint8_t> output_tiramisu_u(input.width()/2, input.height()/2);
    Halide::Buffer<uint8_t> output_tiramisu_v(input.width()/2, input.height()/2);

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        rgbyuv420_tiramisu(input, output_tiramisu_y, output_tiramisu_u, output_tiramisu_v);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rgbyuv420_ref(input, output_ref_y, output_ref_u, output_ref_v);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "rgbyuv420",
               {"  Tiramisu "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output_tiramisu_y, "./build/rgbyuv420_y_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_u, "./build/rgbyuv420_u_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_v, "./build/rgbyuv420_v_tiramisu.png");
    Halide::Tools::save_image(output_ref_y, "./build/rgbyuv420_y_ref.png");
    Halide::Tools::save_image(output_ref_u, "./build/rgbyuv420_u_ref.png");
    Halide::Tools::save_image(output_ref_v, "./build/rgbyuv420_v_ref.png");
    return 0;
}
