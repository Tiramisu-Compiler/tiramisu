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
    tiramisu::function cvtcolor_tiramisu("cvtcolor_tiramisu");

    tiramisu::computation SIZES("{SIZES[i]: 0<=i<=1}", tiramisu::expr(), false, p_int32, &cvtcolor_tiramisu);
    tiramisu::buffer SIZES_b("SIZES_b", {tiramisu::expr(2)}, tiramisu::p_int32, tiramisu::a_input, &cvtcolor_tiramisu);
    SIZES.bind_to(&SIZES_b);
    tiramisu::constant SIZE0("SIZE0", SIZES(0), p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::constant SIZE1("SIZE1", SIZES(1), p_int32, true, NULL, 0, &cvtcolor_tiramisu);

    // Output buffers.
    tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {tiramisu::var("SIZE1"), tiramisu::var("SIZE0")}, tiramisu::p_uint8, tiramisu::a_output, &cvtcolor_tiramisu);

    // Input buffers.
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(3), tiramisu::var("SIZE1"), tiramisu::var("SIZE0")}, tiramisu::p_uint8, tiramisu::a_input, &cvtcolor_tiramisu);
    tiramisu::computation input("[SIZE1, SIZE0]->{input[i2, i1, i0]: (0 <= i2 <= 2) and (0 <= i1 <= (SIZE1 + -1)) and (0 <= i0 <= (SIZE0 + -1))}", expr(), false, tiramisu::p_uint8, &cvtcolor_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");


    // Define loop bounds for dimension "RGB2Gray_s0_y".
    tiramisu::constant RGB2Gray_s0_y_loop_min("RGB2Gray_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::constant RGB2Gray_s0_y_loop_extent("RGB2Gray_s0_y_loop_extent", tiramisu::var("SIZE1"), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);

    // Define loop bounds for dimension "RGB2Gray_s0_x".
    tiramisu::constant RGB2Gray_s0_x_loop_min("RGB2Gray_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::constant RGB2Gray_s0_x_loop_extent("RGB2Gray_s0_x_loop_extent", tiramisu::var("SIZE0"), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::computation RGB2Gray_s0(
        "[RGB2Gray_s0_y_loop_min, RGB2Gray_s0_y_loop_extent, RGB2Gray_s0_x_loop_min, RGB2Gray_s0_x_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_y, RGB2Gray_s0_x]: "
        "(RGB2Gray_s0_y_loop_min <= RGB2Gray_s0_y <= ((RGB2Gray_s0_y_loop_min + RGB2Gray_s0_y_loop_extent) + -1)) and (RGB2Gray_s0_x_loop_min <= RGB2Gray_s0_x <= ((RGB2Gray_s0_x_loop_min + RGB2Gray_s0_x_loop_extent) + -1))}",
	tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32, input(tiramisu::expr((int32_t)2), tiramisu::var("RGB2Gray_s0_y"), tiramisu::var("RGB2Gray_s0_x"))) * tiramisu::expr((uint32_t)1868)) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32, input(tiramisu::expr((int32_t)1), tiramisu::var("RGB2Gray_s0_y"), tiramisu::var("RGB2Gray_s0_x"))) * tiramisu::expr((uint32_t)9617))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint32, input(tiramisu::expr((int32_t)0), tiramisu::var("RGB2Gray_s0_y"), tiramisu::var("RGB2Gray_s0_x"))) * tiramisu::expr((uint32_t)4899))) + tiramisu::expr((uint32_t)8192)) / tiramisu::expr((uint32_t)16384))), true, tiramisu::p_uint8, &cvtcolor_tiramisu);
    RGB2Gray_s0.set_access("{RGB2Gray_s0[RGB2Gray_s0_y, RGB2Gray_s0_x]->buff_RGB2Gray[RGB2Gray_s0_y, RGB2Gray_s0_x]}");

    // Declare vars.
    tiramisu::var RGB2Gray_s0_x("RGB2Gray_s0_x");
    tiramisu::var RGB2Gray_s0_x_v9("RGB2Gray_s0_x_v9");
    tiramisu::var RGB2Gray_s0_x_x("RGB2Gray_s0_x_x");
    tiramisu::var RGB2Gray_s0_y("RGB2Gray_s0_y");

    // Add schedules.
    RGB2Gray_s0.vectorize(RGB2Gray_s0_x, 8);
    RGB2Gray_s0.tag_parallel_level(RGB2Gray_s0_y);

    cvtcolor_tiramisu.add_context_constraints("[RGB2Gray_s0_x_loop_extent, RGB2Gray_s0_x_loop_min]->{: RGB2Gray_s0_x_loop_extent > 1 and RGB2Gray_s0_x_loop_min = 0}");

    cvtcolor_tiramisu.set_arguments({&SIZES_b, &buff_input, &buff_RGB2Gray});
    cvtcolor_tiramisu.gen_time_space_domain();
    cvtcolor_tiramisu.gen_isl_ast();
    cvtcolor_tiramisu.gen_halide_stmt();
    cvtcolor_tiramisu.dump_halide_stmt();
    cvtcolor_tiramisu.gen_halide_obj("build/generated_fct_cvtcolor.o");

    return 0;
}

