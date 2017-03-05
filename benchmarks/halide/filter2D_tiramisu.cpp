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

    tiramisu::function filter2D_tiramisu("filter2D_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int filter2D_extent_2 = SIZE2;
    int filter2D_extent_1 = SIZE1 - 8;
    int filter2D_extent_0 = SIZE0 - 8;
    tiramisu::buffer buff_filter2D("buff_filter2D", 3, {tiramisu::expr(filter2D_extent_2), tiramisu::expr(filter2D_extent_1), tiramisu::expr(filter2D_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &filter2D_tiramisu);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", 3, {tiramisu::expr(input_extent_2), tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_input, &filter2D_tiramisu);
    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &filter2D_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;
    tiramisu::buffer buff_kernel("buff_kernel", 2, {tiramisu::expr(kernel_extent_1), tiramisu::expr(kernel_extent_0)}, tiramisu::p_float32, NULL, tiramisu::a_input, &filter2D_tiramisu);
    tiramisu::computation kernel("[kernel_extent_1, kernel_extent_0]->{kernel[i1, i0]: (0 <= i1 <= (kernel_extent_1 + -1)) and (0 <= i0 <= (kernel_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &filter2D_tiramisu);
    kernel.set_access("{kernel[i1, i0]->buff_kernel[i1, i0]}");


    // Define loop bounds for dimension "filter2D_s0_c".
    tiramisu::constant filter2D_s0_c_loop_min("filter2D_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);
    tiramisu::constant filter2D_s0_c_loop_extent("filter2D_s0_c_loop_extent", tiramisu::expr(filter2D_extent_2), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);

    // Define loop bounds for dimension "filter2D_s0_y".
    tiramisu::constant filter2D_s0_y_loop_min("filter2D_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);
    tiramisu::constant filter2D_s0_y_loop_extent("filter2D_s0_y_loop_extent", tiramisu::expr(filter2D_extent_1), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);

    // Define loop bounds for dimension "filter2D_s0_x".
    tiramisu::constant filter2D_s0_x_loop_min("filter2D_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);
    tiramisu::constant filter2D_s0_x_loop_extent("filter2D_s0_x_loop_extent", tiramisu::expr(filter2D_extent_0), tiramisu::p_int32, true, NULL, 0, &filter2D_tiramisu);
    tiramisu::computation filter2D_s0("[filter2D_s0_c_loop_min, filter2D_s0_c_loop_extent, filter2D_s0_y_loop_min, filter2D_s0_y_loop_extent, filter2D_s0_x_loop_min, filter2D_s0_x_loop_extent]->{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]: "
                        "(filter2D_s0_c_loop_min <= filter2D_s0_c <= ((filter2D_s0_c_loop_min + filter2D_s0_c_loop_extent) + -1)) and (filter2D_s0_y_loop_min <= filter2D_s0_y <= ((filter2D_s0_y_loop_min + filter2D_s0_y_loop_extent) + -1)) and (filter2D_s0_x_loop_min <= filter2D_s0_x <= ((filter2D_s0_x_loop_min + filter2D_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((((((((tiramisu::expr((float)0) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("filter2D_s0_c"), (tiramisu::idx("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::idx("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)2))))), true, tiramisu::p_uint8, &filter2D_tiramisu);
    filter2D_s0.set_access("{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->buff_filter2D[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]}");

    // Define compute level for "filter2D".
    filter2D_s0.set_schedule("[filter2D_s0_c_loop_min, filter2D_s0_c_loop_extent, filter2D_s0_y_loop_min, filter2D_s0_y_loop_extent, filter2D_s0_x_loop_min, filter2D_s0_x_loop_extent]->{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->filter2D_s0[0, 0, filter2D_s0_c, 0, filter2D_s0_y, 0, filter2D_s0_x1, 0, filter2D_s0_x2, 0]: filter2D_s0_x_loop_min <= filter2D_s0_x <= floor((filter2D_s0_x_loop_min+filter2D_s0_x_loop_extent-1)/8)*8 and filter2D_s0_x1 = floor(filter2D_s0_x/8) and filter2D_s0_x2 = filter2D_s0_x%8 and (floor((filter2D_s0_x_loop_min-filter2D_s0_x_loop_extent-1)/8)*8)%8=0; }");
    //   filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->filter2D_s0[filter2D_s0_c, filter2D_s0_y, 1, filter2D_s0_x1, filter2D_s0_x2]: floor((filter2D_s0_x_loop_min+filter2D_s0_x_loop_extent-1)/8)*8 <= filter2D_s0_x <= (filter2D_s0_x_loop_min+filter2D_s0_x_loop_extent-1) and filter2D_s0_x1 = floor(filter2D_s0_x/8) and filter2D_s0_x2 = filter2D_s0_x%8

    // Add schedules.
    filter2D_s0.tag_parallel_level(0);
    filter2D_s0.tag_parallel_level(1);
    filter2D_s0.tag_vector_level(3);

    filter2D_tiramisu.set_arguments({&buff_input, &buff_kernel, &buff_filter2D});
    filter2D_tiramisu.gen_time_processor_domain();
    filter2D_tiramisu.gen_isl_ast();
    filter2D_tiramisu.gen_halide_stmt();
    filter2D_tiramisu.dump_halide_stmt();
    filter2D_tiramisu.gen_halide_obj("build/generated_fct_filter2D.o");

    return 0;
}

