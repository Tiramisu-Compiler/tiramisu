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

    Halide::Image<int16_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Image<uint8_t> output_ref_y(input.width(), input.height());
    Halide::Image<uint8_t> output_ref_u(input.width()/2, input.height()/2);
    Halide::Image<uint8_t> output_ref_v(input.width()/2, input.height()/2);

    Halide::Image<uint8_t> output_coli_y(input.width(), input.height());
    Halide::Image<uint8_t> output_coli_u(input.width()/2, input.height()/2);
    Halide::Image<uint8_t> output_coli_v(input.width()/2, input.height()/2);

    // COLi
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        rgbyuv420_coli(input, output_coli_y, output_coli_u, output_coli_v);
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
               {"  COLi "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output_coli_y, "./build/rgbyuv420_y_coli.png");
    Halide::Tools::save_image(output_coli_u, "./build/rgbyuv420_u_coli.png");
    Halide::Tools::save_image(output_coli_v, "./build/rgbyuv420_v_coli.png");
    Halide::Tools::save_image(output_ref_y, "./build/rgbyuv420_y_ref.png");
    Halide::Tools::save_image(output_ref_u, "./build/rgbyuv420_u_ref.png");
    Halide::Tools::save_image(output_ref_v, "./build/rgbyuv420_v_ref.png");
    return 0;
}
