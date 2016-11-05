#include "wrapper_cvtcolor.h"
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

/*    buffer_t input_buf = {0};
    input_buf.host = (unsigned char *) malloc(NN*NN*NN*sizeof(unsigned char));
    input_buf.stride[0] = 1;
    input_buf.stride[1] = 1;
    input_buf.stride[2] = 1;
    input_buf.extent[0] = NN;
    input_buf.extent[1] = NN;
    input_buf.stride[2] = 3;
    input_buf.min[0] = 0;
    input_buf.min[1] = 0;
    input_buf.elem_size = 1;*/


    Halide::Image<uint8_t> output1(input.width(), input.height(), 3);
    Halide::Image<uint8_t> output2(input.width(), input.height(), 3);

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        cvtcolor_coli(input, output1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // COLi
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        cvtcolor_ref(input, output2);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "cvtcolor",
               {"  COLi "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output1, "./build/cvtcolor_coli.png");
    Halide::Tools::save_image(output2, "./build/cvtcolor_ref.png");

    return 0;
}
