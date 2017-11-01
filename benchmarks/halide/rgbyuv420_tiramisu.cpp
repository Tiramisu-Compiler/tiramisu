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

    tiramisu::function rgbyuv420("rgbyuv420_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int y_part_extent_1 = SIZE1;
    int y_part_extent_0 = SIZE0;
    tiramisu::buffer buff_y_part("buff_y_part", {tiramisu::expr(y_part_extent_1), tiramisu::expr(y_part_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &rgbyuv420);
    int u_part_extent_1 = SIZE1;
    int u_part_extent_0 = SIZE0;
    tiramisu::buffer buff_u_part("buff_u_part", {tiramisu::expr(u_part_extent_1), tiramisu::expr(u_part_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &rgbyuv420);
    int v_part_extent_1 = SIZE1;
    int v_part_extent_0 = SIZE0;
    tiramisu::buffer buff_v_part("buff_v_part", {tiramisu::expr(v_part_extent_1), tiramisu::expr(v_part_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &rgbyuv420);

    // Input buffers.
    int p0_extent_2 = SIZE2;
    int p0_extent_1 = SIZE1;
    int p0_extent_0 = SIZE0;
    tiramisu::buffer buff_p0("buff_p0", {tiramisu::expr(p0_extent_2), tiramisu::expr(p0_extent_1), tiramisu::expr(p0_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &rgbyuv420);
    tiramisu::computation p0("[p0_extent_2, p0_extent_1, p0_extent_0]->{p0[i2, i1, i0]: (0 <= i2 <= (p0_extent_2 + -1)) and (0 <= i1 <= (p0_extent_1 + -1)) and (0 <= i0 <= (p0_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &rgbyuv420);
    p0.set_access("{p0[i2, i1, i0]->buff_p0[i2, i1, i0]}");


    // Define loop bounds for dimension "y_part_s0_y".
    tiramisu::constant y_part_s0_y_loop_min("y_part_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant y_part_s0_y_loop_extent("y_part_s0_y_loop_extent", tiramisu::expr(y_part_extent_1), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "y_part_s0_x".
    tiramisu::constant y_part_s0_x_loop_min("y_part_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant y_part_s0_x_loop_extent("y_part_s0_x_loop_extent", tiramisu::expr(y_part_extent_0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::computation y_part_s0("[y_part_s0_y_loop_min, y_part_s0_y_loop_extent, y_part_s0_x_loop_min, y_part_s0_x_loop_extent]->{y_part_s0[y_part_s0_y, y_part_s0_x]: "
                        "(y_part_s0_y_loop_min <= y_part_s0_y <= ((y_part_s0_y_loop_min + y_part_s0_y_loop_extent) + -1)) and (y_part_s0_x_loop_min <= y_part_s0_x <= ((y_part_s0_x_loop_min + y_part_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, ((((((tiramisu::expr((uint8_t)66)*p0(tiramisu::expr((int32_t)0), tiramisu::var("y_part_s0_y"), tiramisu::var("y_part_s0_x"))) + (tiramisu::expr((uint8_t)129)*p0(tiramisu::expr((int32_t)1), tiramisu::var("y_part_s0_y"), tiramisu::var("y_part_s0_x")))) + (tiramisu::expr((uint8_t)25)*p0(tiramisu::expr((int32_t)2), tiramisu::var("y_part_s0_y"), tiramisu::var("y_part_s0_x")))) + tiramisu::expr((uint8_t)128)) >> tiramisu::expr((uint8_t)8)) + tiramisu::expr((uint8_t)16))), true, tiramisu::p_uint8, &rgbyuv420);
    y_part_s0.set_access("{y_part_s0[y_part_s0_y, y_part_s0_x]->buff_y_part[y_part_s0_y, y_part_s0_x]}");

    // Define loop bounds for dimension "u_part_s0_y".
    tiramisu::constant u_part_s0_y_loop_min("u_part_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant u_part_s0_y_loop_extent("u_part_s0_y_loop_extent", tiramisu::expr(u_part_extent_1), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "u_part_s0_x".
    tiramisu::constant u_part_s0_x_loop_min("u_part_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant u_part_s0_x_loop_extent("u_part_s0_x_loop_extent", tiramisu::expr(u_part_extent_0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::computation u_part_s0("[u_part_s0_y_loop_min, u_part_s0_y_loop_extent, u_part_s0_x_loop_min, u_part_s0_x_loop_extent]->{u_part_s0[u_part_s0_y, u_part_s0_x]: "
                        "(u_part_s0_y_loop_min <= u_part_s0_y <= ((u_part_s0_y_loop_min + u_part_s0_y_loop_extent) + -1)) and (u_part_s0_x_loop_min <= u_part_s0_x <= ((u_part_s0_x_loop_min + u_part_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(), true, tiramisu::p_uint8, &rgbyuv420);
    tiramisu::constant t0("t0", (tiramisu::var("u_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    tiramisu::constant t1("t1", (tiramisu::var("u_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    tiramisu::constant t2("t2", (tiramisu::var("u_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    tiramisu::constant t3("t3", (tiramisu::var("u_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    tiramisu::constant t4("t4", (tiramisu::var("u_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    tiramisu::constant t5("t5", (tiramisu::var("u_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    u_part_s0.set_expression(tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, ((((((tiramisu::expr((uint8_t)38)*p0(tiramisu::expr((int32_t)0), t0, t1)) - (tiramisu::expr((uint8_t)74)*p0(tiramisu::expr((int32_t)1), t2, t3))) + (tiramisu::expr((uint8_t)112)*p0(tiramisu::expr((int32_t)2), t4, t5))) + tiramisu::expr((uint8_t)128)) >> tiramisu::expr((uint8_t)8)) + tiramisu::expr((uint8_t)128))));
    u_part_s0.set_access("{u_part_s0[u_part_s0_y, u_part_s0_x]->buff_u_part[u_part_s0_y, u_part_s0_x]}");

    // Define loop bounds for dimension "v_part_s0_y".
    tiramisu::constant v_part_s0_y_loop_min("v_part_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant v_part_s0_y_loop_extent("v_part_s0_y_loop_extent", tiramisu::expr(v_part_extent_1), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "v_part_s0_x".
    tiramisu::constant v_part_s0_x_loop_min("v_part_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::constant v_part_s0_x_loop_extent("v_part_s0_x_loop_extent", tiramisu::expr(v_part_extent_0), tiramisu::p_int32, true, NULL, 0, &rgbyuv420);
    tiramisu::computation v_part_s0("[v_part_s0_y_loop_min, v_part_s0_y_loop_extent, v_part_s0_x_loop_min, v_part_s0_x_loop_extent]->{v_part_s0[v_part_s0_y, v_part_s0_x]: "
                        "(v_part_s0_y_loop_min <= v_part_s0_y <= ((v_part_s0_y_loop_min + v_part_s0_y_loop_extent) + -1)) and (v_part_s0_x_loop_min <= v_part_s0_x <= ((v_part_s0_x_loop_min + v_part_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(), true, tiramisu::p_uint8, &rgbyuv420);
    tiramisu::constant t6("t6", (tiramisu::var("v_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    tiramisu::constant t7("t7", (tiramisu::var("v_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    tiramisu::constant t8("t8", (tiramisu::var("v_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    tiramisu::constant t9("t9", (tiramisu::var("v_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    tiramisu::constant t10("t10", (tiramisu::var("v_part_s0_y")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    tiramisu::constant t11("t11", (tiramisu::var("v_part_s0_x")*tiramisu::expr((int32_t)2)), tiramisu::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    v_part_s0.set_expression(tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, ((((((tiramisu::expr((uint8_t)112)*p0(tiramisu::expr((int32_t)0), t6, t7)) - (tiramisu::expr((uint8_t)94)*p0(tiramisu::expr((int32_t)1), t8, t9))) - (tiramisu::expr((uint8_t)18)*p0(tiramisu::expr((int32_t)2), t10, t11))) + tiramisu::expr((uint8_t)128)) >> tiramisu::expr((uint8_t)8)) + tiramisu::expr((uint8_t)128))));
    v_part_s0.set_access("{v_part_s0[v_part_s0_y, v_part_s0_x]->buff_v_part[v_part_s0_y, v_part_s0_x]}");

    tiramisu::var y_part_s0_x("y_part_s0_x");
    tiramisu::var u_part_s0_x("u_part_s0_x");

    rgbyuv420.add_context_constraints("[v_part_s0_y_loop_min, u_part_s0_y_loop_min, y_part_s0_y_loop_min, v_part_s0_y_loop_extent, u_part_s0_y_loop_extent, y_part_s0_y_loop_extent, v_part_s0_x_loop_min, u_part_s0_x_loop_min, y_part_s0_x_loop_min, v_part_s0_x_loop_extent, u_part_s0_x_loop_extent, y_part_s0_x_loop_extent]->{: v_part_s0_x_loop_min = 0 and u_part_s0_x_loop_min = 0 and y_part_s0_x_loop_min = 0 and v_part_s0_x_loop_extent > 1 and u_part_s0_x_loop_extent > 1 and y_part_s0_x_loop_extent > 1 and v_part_s0_x_loop_extent = y_part_s0_x_loop_extent and v_part_s0_x_loop_extent = u_part_s0_x_loop_extent and v_part_s0_y_loop_min = 0 and u_part_s0_y_loop_min = 0 and y_part_s0_y_loop_min = 0 and v_part_s0_y_loop_extent > 1 and u_part_s0_y_loop_extent > 1 and y_part_s0_y_loop_extent > 1 and v_part_s0_y_loop_extent = y_part_s0_y_loop_extent and v_part_s0_y_loop_extent = u_part_s0_y_loop_extent }");

    // Define compute level for "v_part".
    t0.after(y_part_s0, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    t1.after(t0, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    t2.after(t1, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    t3.after(t2, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    t4.after(t3, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    t5.after(t4, y_part_s0.get_loop_level_number_from_dimension_name("y_part_s0_x"));
    u_part_s0.after(t5, u_part_s0_x);
    t6.after(u_part_s0, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    t7.after(t6, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    t8.after(t7, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    t9.after(t8, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    t10.after(t9, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    t11.after(t10, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));
    v_part_s0.after(t11, u_part_s0.get_loop_level_number_from_dimension_name("u_part_s0_x"));

    // Add schedules.
    rgbyuv420.set_arguments({&buff_p0, &buff_u_part, &buff_v_part, &buff_y_part});
    rgbyuv420.gen_time_space_domain();
    rgbyuv420.gen_isl_ast();
    rgbyuv420.gen_halide_stmt();
    rgbyuv420.dump_halide_stmt();
    rgbyuv420.gen_halide_obj("build/generated_fct_rgbyuv420.o");

    return 0;
}

