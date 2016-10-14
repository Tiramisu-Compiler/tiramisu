#include "wrapper_blurxy.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "coli/utils.h"
#include <cstdlib>
#include <iostream>

#define NN 1000

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Image<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Image<uint8_t> output1(input.width()-8, input.height()-2);
    Halide::Image<uint8_t> output2(input.width()-8, input.height()-2);

    // Warm up
    for (int i=0; i<10; i++)
       blurxy_coli(input, output1);

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        blurxy_coli(input, output1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // COLi
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        blurxy_ref(input, output2);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "blurxy",
               {"  COLi "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output1, "./build/blurxy.png");

    return 0;
}
