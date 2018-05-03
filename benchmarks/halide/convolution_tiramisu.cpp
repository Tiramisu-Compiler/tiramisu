#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_default_data_type(p_int32);

    tiramisu::function convolution_tiramisu("convolution_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int convolution_extent_2 = SIZE2;
    int convolution_extent_1 = SIZE1 - 8;
    int convolution_extent_0 = SIZE0 - 8;
    tiramisu::buffer buff_convolution("buff_convolution", {tiramisu::expr(convolution_extent_2), tiramisu::expr(convolution_extent_1), tiramisu::expr(convolution_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &convolution_tiramisu);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_2), tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &convolution_tiramisu);
    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &convolution_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;
    tiramisu::buffer buff_kernel("buff_kernel", {tiramisu::expr(kernel_extent_1), tiramisu::expr(kernel_extent_0)}, tiramisu::p_float32, tiramisu::a_input, &convolution_tiramisu);
    tiramisu::computation kernel("[kernel_extent_1, kernel_extent_0]->{kernel[i1, i0]: (0 <= i1 <= (kernel_extent_1 + -1)) and (0 <= i0 <= (kernel_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &convolution_tiramisu);
    kernel.set_access("{kernel[i1, i0]->buff_kernel[i1, i0]}");


    // Define loop bounds for dimension "convolution_s0_c".
    tiramisu::constant convolution_s0_c_loop_min("convolution_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);
    tiramisu::constant convolution_s0_c_loop_extent("convolution_s0_c_loop_extent", tiramisu::expr(convolution_extent_2), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);

    // Define loop bounds for dimension "convolution_s0_y".
    tiramisu::constant convolution_s0_y_loop_min("convolution_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);
    tiramisu::constant convolution_s0_y_loop_extent("convolution_s0_y_loop_extent", tiramisu::expr(convolution_extent_1), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);

    // Define loop bounds for dimension "convolution_s0_x".
    tiramisu::constant convolution_s0_x_loop_min("convolution_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);
    tiramisu::constant convolution_s0_x_loop_extent("convolution_s0_x_loop_extent", tiramisu::expr(convolution_extent_0), tiramisu::p_int32, true, NULL, 0, &convolution_tiramisu);
    tiramisu::computation convolution_s0("[convolution_s0_c_loop_min, convolution_s0_c_loop_extent, convolution_s0_y_loop_min, convolution_s0_y_loop_extent, convolution_s0_x_loop_min, convolution_s0_x_loop_extent]->{convolution_s0[convolution_s0_c, convolution_s0_y, convolution_s0_x]: "
                        "(convolution_s0_c_loop_min <= convolution_s0_c <= ((convolution_s0_c_loop_min + convolution_s0_c_loop_extent) + -1)) and (convolution_s0_y_loop_min <= convolution_s0_y <= ((convolution_s0_y_loop_min + convolution_s0_y_loop_extent) + -1)) and (convolution_s0_x_loop_min <= convolution_s0_x <= ((convolution_s0_x_loop_min + convolution_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((((((((tiramisu::expr((float)0) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("convolution_s0_c"), (tiramisu::var("convolution_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("convolution_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)2))))), true, tiramisu::p_uint8, &convolution_tiramisu);
    convolution_s0.set_access("{convolution_s0[convolution_s0_c, convolution_s0_y, convolution_s0_x]->buff_convolution[convolution_s0_c, convolution_s0_y, convolution_s0_x]}");

    // Add schedules.
    convolution_s0.tag_parallel_level(tiramisu::var("convolution_s0_c"));
    convolution_s0.tag_parallel_level(tiramisu::var("convolution_s0_y"));
    convolution_s0.vectorize(tiramisu::var("convolution_s0_x"), 8);

    convolution_tiramisu.set_arguments({&buff_input, &buff_kernel, &buff_convolution});
    convolution_tiramisu.gen_time_space_domain();
    convolution_tiramisu.gen_isl_ast();
    convolution_tiramisu.gen_halide_stmt();
    convolution_tiramisu.dump_halide_stmt();
    convolution_tiramisu.gen_halide_obj("build/generated_fct_convolution.o");

    return 0;
}

