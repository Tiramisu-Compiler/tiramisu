#include "wrapper_test_14.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char **)
{
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./utils/images/rgb.png");
    Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./utils/images/reference_blurxy.png");
    Halide::Buffer<uint8_t> output(input.width() - 8, input.height() - 8, input.channels());

    blurxy_tiramisu_test(input.raw_buffer(), output.raw_buffer());

    //TODO(psuriana): Fix this test
    // compare_buffers("blurxy", output, ref);
    Halide::Tools::save_image(output, "./build/blurxy_tiramisu_test.png");

    return 0;
}
