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

    tiramisu::function heat2d_tiramisu("heat2d_tiramisu");

    // Input params.
    float alpha = 0.3;
    float beta = 0.4;

    int SIZE0 = 10000;
    int SIZE1 = 10000;

    // Output buffers.
    int heat2d_extent_1 = SIZE1;
    int heat2d_extent_0 = SIZE0;
    tiramisu::buffer buff_heat2d("buff_heat2d", {tiramisu::expr(heat2d_extent_1), tiramisu::expr(heat2d_extent_0)}, tiramisu::p_float32, tiramisu::a_output, &heat2d_tiramisu);

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_float32, tiramisu::a_input, &heat2d_tiramisu);
    tiramisu::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &heat2d_tiramisu);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "heat2d_s0_y".
    tiramisu::constant heat2d_s0_y_loop_min("heat2d_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::constant heat2d_s0_y_loop_extent("heat2d_s0_y_loop_extent", tiramisu::expr(heat2d_extent_1), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);

    // Define loop bounds for dimension "heat2d_s0_x".
    tiramisu::constant heat2d_s0_x_loop_min("heat2d_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::constant heat2d_s0_x_loop_extent("heat2d_s0_x_loop_extent", tiramisu::expr(heat2d_extent_0), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::computation heat2d_s0("[heat2d_s0_y_loop_min, heat2d_s0_y_loop_extent, heat2d_s0_x_loop_min, heat2d_s0_x_loop_extent]->{heat2d_s0[heat2d_s0_y, heat2d_s0_x]: "
                        "(heat2d_s0_y_loop_min <= heat2d_s0_y <= ((heat2d_s0_y_loop_min + heat2d_s0_y_loop_extent) + -1)) and (heat2d_s0_x_loop_min <= heat2d_s0_x <= ((heat2d_s0_x_loop_min + heat2d_s0_x_loop_extent) + -1))}",
                        tiramisu::expr((float)0), true, tiramisu::p_float32, &heat2d_tiramisu);
    heat2d_s0.set_access("{heat2d_s0[heat2d_s0_y, heat2d_s0_x]->buff_heat2d[heat2d_s0_y, heat2d_s0_x]}");

    // Define loop bounds for dimension "heat2d_s1_r__y".
    tiramisu::constant heat2d_s1_r__y_loop_min("heat2d_s1_r__y_loop_min", tiramisu::expr((int32_t)1), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::constant heat2d_s1_r__y_loop_extent("heat2d_s1_r__y_loop_extent", (tiramisu::expr(input_extent_1) + tiramisu::expr((int32_t)-2)), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);

    // Define loop bounds for dimension "heat2d_s1_r__x".
    tiramisu::constant heat2d_s1_r__x_loop_min("heat2d_s1_r__x_loop_min", tiramisu::expr((int32_t)1), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::constant heat2d_s1_r__x_loop_extent("heat2d_s1_r__x_loop_extent", (tiramisu::expr(input_extent_0) + tiramisu::expr((int32_t)-2)), tiramisu::p_int32, true, NULL, 0, &heat2d_tiramisu);
    tiramisu::computation heat2d_s1(
        "[heat2d_s1_r__y_loop_min, heat2d_s1_r__y_loop_extent, heat2d_s1_r__x_loop_min, heat2d_s1_r__x_loop_extent]->{heat2d_s1[heat2d_s1_r__y, heat2d_s1_r__x]: "
        "(heat2d_s1_r__y_loop_min <= heat2d_s1_r__y <= ((heat2d_s1_r__y_loop_min + heat2d_s1_r__y_loop_extent) + -1)) and (heat2d_s1_r__x_loop_min <= heat2d_s1_r__x <= ((heat2d_s1_r__x_loop_min + heat2d_s1_r__x_loop_extent) + -1))}",
        tiramisu::expr(), true, tiramisu::p_float32, &heat2d_tiramisu);
    heat2d_s1.set_expression(((tiramisu::expr(alpha) * input(tiramisu::var("heat2d_s1_r__y"), tiramisu::var("heat2d_s1_r__x"))) + (tiramisu::expr(beta) * (((input(tiramisu::var("heat2d_s1_r__y"), (tiramisu::var("heat2d_s1_r__x") + tiramisu::expr((int32_t)1))) + input(tiramisu::var("heat2d_s1_r__y"), (tiramisu::var("heat2d_s1_r__x") - tiramisu::expr((int32_t)1)))) + input((tiramisu::var("heat2d_s1_r__y") + tiramisu::expr((int32_t)1)), tiramisu::var("heat2d_s1_r__x"))) + input((tiramisu::var("heat2d_s1_r__y") - tiramisu::expr((int32_t)1)), tiramisu::var("heat2d_s1_r__x"))))));
    heat2d_s1.set_access("{heat2d_s1[heat2d_s1_r__y, heat2d_s1_r__x]->buff_heat2d[heat2d_s1_r__y, heat2d_s1_r__x]}");

    // Define compute level for "heat2d".
    heat2d_s1.after(heat2d_s0, computation::root);

    // Declare vars.
    tiramisu::var heat2d_s0_x("heat2d_s0_x");
    tiramisu::var heat2d_s0_y("heat2d_s0_y");
    tiramisu::var heat2d_s1_r__x("heat2d_s1_r__x");
    tiramisu::var heat2d_s1_r__y("heat2d_s1_r__y");

    // Add schedules.
    heat2d_s0.tag_parallel_level(heat2d_s0_y);
    //heat2d_s0.vectorize(heat2d_s0_x, 8);
    heat2d_s1.tag_parallel_level(heat2d_s1_r__y);
    //heat2d_s1.vectorize(heat2d_s1_r__x, 8);

    heat2d_tiramisu.set_arguments({&buff_input, &buff_heat2d});
    heat2d_tiramisu.gen_time_space_domain();
    heat2d_tiramisu.gen_isl_ast();
    heat2d_tiramisu.gen_halide_stmt();
    heat2d_tiramisu.dump_halide_stmt();
    heat2d_tiramisu.gen_halide_obj("build/generated_fct_heat2d.o");

    return 0;
}
