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

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);

    tiramisu::function resize_tiramisu("resize_tiramisu");

    // Input params.
    int32_t resampled_cols = 5;
    int32_t resampled_rows = 5;

    // Output buffers.
    int resampled_extent_1 = SIZE1;
    int resampled_extent_0 = SIZE0;
    tiramisu::buffer buff_resampled("buff_resampled", {tiramisu::expr(resampled_extent_1), tiramisu::expr(resampled_extent_0)}, tiramisu::p_float32, tiramisu::a_output, &resize_tiramisu);

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &resize_tiramisu);
    tiramisu::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", tiramisu::expr(), false, tiramisu::p_uint8, &resize_tiramisu);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "resampled_s0_y".
    tiramisu::constant resampled_s0_y_loop_min("resampled_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &resize_tiramisu);
    tiramisu::constant resampled_s0_y_loop_extent("resampled_s0_y_loop_extent", tiramisu::expr((int32_t)resampled_extent_1), tiramisu::p_int32, true, NULL, 0, &resize_tiramisu);

    // Define loop bounds for dimension "resampled_s0_x".
    tiramisu::constant resampled_s0_x_loop_min("resampled_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &resize_tiramisu);
    tiramisu::constant resampled_s0_x_loop_extent("resampled_s0_x_loop_extent", tiramisu::expr((int32_t)resampled_extent_0), tiramisu::p_int32, true, NULL, 0, &resize_tiramisu);
    tiramisu::computation resampled_s0(
        "[resampled_s0_y_loop_min, resampled_s0_y_loop_extent, resampled_s0_x_loop_min, resampled_s0_x_loop_extent]->{resampled_s0[resampled_s0_y, resampled_s0_x]: "
        "(resampled_s0_y_loop_min <= resampled_s0_y <= ((resampled_s0_y_loop_min + resampled_s0_y_loop_extent) + -1)) and (resampled_s0_x_loop_min <= resampled_s0_x <= ((resampled_s0_x_loop_min + resampled_s0_x_loop_extent) + -1))}",
        tiramisu::expr(), true, tiramisu::p_float32, &resize_tiramisu);
    tiramisu::constant t37("t37", tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))), tiramisu::p_int32, false, &resampled_s0, 1, &resize_tiramisu);
    tiramisu::constant t38("t38", tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))), tiramisu::p_int32, false, &resampled_s0, 1, &resize_tiramisu);
    tiramisu::constant t39("t39", tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))), tiramisu::p_int32, false, &resampled_s0, 1, &resize_tiramisu);
    tiramisu::constant t40("t40", tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))), tiramisu::p_int32, false, &resampled_s0, 1, &resize_tiramisu);
    resampled_s0.set_expression(((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t37(0, 0), t38(0, 0))) * (tiramisu::expr((float)1) - (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t39(0, 0), (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((int32_t)1)))) * (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))) * (tiramisu::expr((float)1) - (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))) + (((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input((tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((int32_t)1)), t40(0, 0))) * (tiramisu::expr((float)1) - (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input((tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((int32_t)1)), (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((int32_t)1)))) * (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_y")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_1))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_rows))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))) * (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) - tiramisu::expr(o_floor, ((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("resampled_s0_x")) + tiramisu::expr((float)0.5)) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(input_extent_0))) / tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::expr(resampled_cols))) + tiramisu::expr((float)-0.5)))) + tiramisu::expr((float)-0.5)))));
    resampled_s0.set_access("{resampled_s0[resampled_s0_y, resampled_s0_x]->buff_resampled[resampled_s0_y, resampled_s0_x]}");

    // Define compute level for "resampled".

    // Declare vars.
    tiramisu::var resampled_s0_x("resampled_s0_x");
    tiramisu::var resampled_s0_x_v8("resampled_s0_x_v8");
    tiramisu::var resampled_s0_x_x("resampled_s0_x_x");
    tiramisu::var resampled_s0_y("resampled_s0_y");

    // Add schedules.
    resampled_s0.vectorize(resampled_s0_x, 8);
    resampled_s0.tag_parallel_level(resampled_s0_y);

    resize_tiramisu.set_arguments({&buff_input, &buff_resampled});
    resize_tiramisu.gen_time_space_domain();
    resize_tiramisu.gen_isl_ast();
    resize_tiramisu.gen_halide_stmt();
    resize_tiramisu.dump_halide_stmt();
    resize_tiramisu.gen_halide_obj("build/generated_fct_resize_tiramisu.o");

    return 0;
}
