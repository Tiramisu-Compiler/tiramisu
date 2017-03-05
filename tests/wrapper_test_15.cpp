#include "wrapper_test_15.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");
    Halide::Buffer<uint8_t> output1(input.width()-8, input.height()-8, input.channels());

    auto start1 = std::chrono::high_resolution_clock::now();
    blurxy_tiramisu_test(input.raw_buffer(), output1.raw_buffer());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
    duration_vector_1.push_back(duration1);

    Halide::Tools::save_image(output1, "./build/blurxy_tiramisu_test.png");

    return 0;
}
