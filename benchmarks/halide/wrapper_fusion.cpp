#include "wrapper_fusion.h"
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

    Halide::Image<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Image<uint8_t> output_ref_f(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_ref_g(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_ref_h(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_ref_k(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_tiramisu_f(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_tiramisu_g(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_tiramisu_h(input.width(), input.height(), input.channels());
    Halide::Image<uint8_t> output_tiramisu_k(input.width(), input.height(), input.channels());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        fusion_tiramisu(input, output_tiramisu_f, output_tiramisu_g, output_tiramisu_h, output_tiramisu_k);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        fusion_ref(input, output_ref_f, output_ref_g, output_ref_h, output_ref_k);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "fusion",
               {"  Tiramisu "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output_tiramisu_f, "./build/fusion_f_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_g, "./build/fusion_g_tiramisu.png");
    Halide::Tools::save_image(output_ref_f, "./build/fusion_f_ref.png");
    Halide::Tools::save_image(output_ref_g, "./build/fusion_g_ref.png");

    return 0;
}
