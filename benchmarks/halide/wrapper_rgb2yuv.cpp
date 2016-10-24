#include "wrapper_blurxy.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "coli/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Image<uint32_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Image<uint32_t> output1(input.width(), input.height());
    Halide::Image<uint32_t> output2(input.width(), input.height());

    // Warm up
    rgb2yuv_coli(input, output1);

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        rgb2yuv_coli(input, output1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // COLi
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rgb2yuv_ref(input, output2);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "rgb2yuv",
               {"  COLi "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

   //compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output1, "./build/rgb2yuv_coli.png");
    Halide::Tools::save_image(output2, "./build/rgb2yuv_ref.png");

    return 0;
}
