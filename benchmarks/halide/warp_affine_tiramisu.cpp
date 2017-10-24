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

    tiramisu::function affine_tiramisu("affine_tiramisu");

    // Input params.
    float a00__0 = 0.1;
    float a01__0 = 0.1;
    float a10__0 = 0.1;
    float a11__0 = 0.1;
    float b00__0 = 0.1;
    float b10__0 = 0.1;

    // Output buffers.
    int affine_extent_1 = SIZE1;
    int affine_extent_0 = SIZE0;
    tiramisu::buffer buff_affine("buff_affine", {tiramisu::expr(affine_extent_1), tiramisu::expr(affine_extent_0)}, tiramisu::p_float32, tiramisu::a_output, &affine_tiramisu);

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &affine_tiramisu);
    tiramisu::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &affine_tiramisu);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "affine_s0_y".
    tiramisu::constant affine_s0_y_loop_min("affine_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &affine_tiramisu);
    tiramisu::constant affine_s0_y_loop_extent("affine_s0_y_loop_extent", tiramisu::expr((int32_t)affine_extent_1), tiramisu::p_int32, true, NULL, 0, &affine_tiramisu);

    // Define loop bounds for dimension "affine_s0_x".
    tiramisu::constant affine_s0_x_loop_min("affine_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &affine_tiramisu);
    tiramisu::constant affine_s0_x_loop_extent("affine_s0_x_loop_extent", tiramisu::expr((int32_t)affine_extent_0), tiramisu::p_int32, true, NULL, 0, &affine_tiramisu);
    tiramisu::computation affine_s0(
        "[affine_s0_y_loop_min, affine_s0_y_loop_extent, affine_s0_x_loop_min, affine_s0_x_loop_extent]->{affine_s0[affine_s0_y, affine_s0_x]: "
        "(affine_s0_y_loop_min <= affine_s0_y <= ((affine_s0_y_loop_min + affine_s0_y_loop_extent) + -1)) and (affine_s0_x_loop_min <= affine_s0_x <= ((affine_s0_x_loop_min + affine_s0_x_loop_extent) + -1))}",
        tiramisu::expr(), true, tiramisu::p_float32, &affine_tiramisu);
    tiramisu::constant t57("t57", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t58("t58", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t59("t59", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t60("t60", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t61("t61", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t62("t62", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t63("t63", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    tiramisu::constant t64("t64", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_tiramisu);
    affine_s0.set_expression(((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t57(0), t58(0))) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t59(0), t60(0))) * ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))))) + (((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t61(0), t62(0))) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t63(0), t64(0))) * ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) * ((((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))))));
    affine_s0.set_access("{affine_s0[affine_s0_y, affine_s0_x]->buff_affine[affine_s0_y, affine_s0_x]}");

    // Define compute level for "affine".

    // Declare vars.
    tiramisu::var affine_s0_x("affine_s0_x");
    tiramisu::var affine_s0_y("affine_s0_y");

    // Add schedules.
    affine_s0.tag_parallel_level(affine_s0_y);
    affine_s0.vectorize(affine_s0_x, 8);

    affine_tiramisu.set_arguments({&buff_input, &buff_affine});
    affine_tiramisu.gen_time_space_domain();
    affine_tiramisu.gen_isl_ast();
    affine_tiramisu.gen_halide_stmt();
    affine_tiramisu.dump_halide_stmt();
    affine_tiramisu.gen_halide_obj("build/generated_fct_affine_tiramisu.o");

    return 0;
}
