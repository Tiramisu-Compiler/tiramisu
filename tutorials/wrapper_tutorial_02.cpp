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
    Halide::Buffer<uint8_t> output_buf(image.extent(0), image.extent(1));

    // The blurxy takes a halide_buffer_t * as argument, when "image"
    // is passed, its buffer is actually extracted and passed
    // to the function (c++ operator overloading).
    blurxy(image.raw_buffer(), output_buf.raw_buffer());

    // TODO(psuriana): not sure why we have to copy the output to image, then
    // write it to file, as opposed to write the output to file directly.
    copy_buffer(output_buf, image);
    Halide::Tools::save_image(image, "./build/tutorial_02.png");

   return 0;
}
