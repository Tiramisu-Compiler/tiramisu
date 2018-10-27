#include <tiramisu/tiramisu.h>

#include <Halide.h>
#include "halide_image_io.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("convolution_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./utils/images/rgb.png");
    constant SIZE0("SIZE0", in_image.extent(0));
    constant SIZE1("SIZE1", in_image.extent(1));
    constant SIZE2("SIZE2", in_image.extent(2));

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;
    var i2("i2", 0, SIZE2), i1("i1", 0, SIZE1), i0("i0", 0, SIZE0);
    var k0("k0", 0, kernel_extent_0), k1("k1", 0, kernel_extent_1);
    var c("c", 0, SIZE2), y("y", 0, SIZE1-8), x("x", 0, SIZE0-8);

    input     in("in", {i2, i1, i0}, p_uint8);
    input kernel("kernel", {k1, k0}, p_float32);

    computation conv("conv", {c, y, x}, cast(p_uint8, (cast(p_float32, cast(p_float32, in(c, y,     x    ))*kernel(0, 0)) +
						       cast(p_float32, cast(p_float32, in(c, y,     x + 1))*kernel(0, 1)) +
						       cast(p_float32, cast(p_float32, in(c, y,     x + 2))*kernel(0, 2)) +
						       cast(p_float32, cast(p_float32, in(c, y + 1, x    ))*kernel(1, 0)) +
						       cast(p_float32, cast(p_float32, in(c, y + 1, x + 1))*kernel(1, 1)) +
						       cast(p_float32, cast(p_float32, in(c, y + 1, x + 2))*kernel(1, 2)) +
						       cast(p_float32, cast(p_float32, in(c, y + 2, x    ))*kernel(2, 0)) +
						       cast(p_float32, cast(p_float32, in(c, y + 2, x + 1))*kernel(2, 1)) +
						       cast(p_float32, cast(p_float32, in(c, y + 2, x + 2))*kernel(2, 2))
						       )));

    // Add schedules.
    conv.parallelize(c);
    conv.vectorize(x, 8);

    // Buffers.
    buffer buff_input("buff_input", {SIZE2, SIZE1, SIZE0}, p_uint8, a_input);
    buffer buff_kernel("buff_kernel", {kernel_extent_1, kernel_extent_0}, p_float32, a_input);
    buffer buff_convolution("buff_convolution", {SIZE2, SIZE1-8, SIZE0-8}, p_uint8, a_output);
    in.store_in(&buff_input);
    conv.store_in(&buff_convolution);
    kernel.store_in(&buff_kernel);

    tiramisu::codegen({&buff_input, &buff_kernel, &buff_convolution} ,"build/generated_fct_convolution.o");

    return 0;
}

