#include "Halide.h"
#include "wrapper_tutorial_02.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

#define NN 10

int main(int, char**)
{
    Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");

    buffer_t output_buf = {0};
    output_buf.host = (unsigned char *) malloc(image.extent(0)*image.extent(1)*sizeof(unsigned char));
    output_buf.stride[0] = 1;
    output_buf.stride[1] = image.extent(0);
    output_buf.extent[0] = image.extent(0);
    output_buf.extent[1] = image.extent(1);
    output_buf.min[0] = 0;
    output_buf.min[1] = 0;
    output_buf.elem_size = 1;

    // The blurxy takes a buffer_t * argument, when "image"
    // is passed, its buffer is actually extracted and passed
    // to the function (c++ operator overloading).
    blurxy(image.raw_buffer(), &output_buf);

    copy_2D_buffer(image.data(), image.extent(0), image.extent(1), output_buf.host);

    Halide::Tools::save_image(image, "./build/tutorial_02.png");

   return 0;
}
