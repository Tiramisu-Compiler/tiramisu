#include "wrapper_test_15.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    typedef std::chrono::duration<double, std::milli> Duration;

    std::vector<Duration> duration_vector;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./utils/images/rgb.png");
    Halide::Buffer<uint8_t> output(input.width()-8, input.height()-8, input.channels());

    auto start = std::chrono::high_resolution_clock::now();
    blurxy_tiramisu_test(input.raw_buffer(), output.raw_buffer());
    auto end = std::chrono::high_resolution_clock::now();
    Duration duration = end - start;
    duration_vector.push_back(duration);

    Halide::Tools::save_image(output, "./blurxy_tiramisu_test.png");

    return 0;
}
