#include "wrapper_test_14.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");
    Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./images/reference_blurxy.png");
    Halide::Buffer<uint8_t> output1(input.width()-8, input.height()-8, input.channels());

    blurxy_tiramisu_test(input.raw_buffer(), output1.raw_buffer());

    compare_2_2D_arrays("blurxy", output1.data(), ref.data(), input.width()-8, input.height()-8);
    Halide::Tools::save_image(output1, "./build/blurxy_tiramisu_test.png");

    return 0;
}
