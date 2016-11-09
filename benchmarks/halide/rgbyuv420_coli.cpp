#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    coli::function rgbyuv420("rgbyuv420_coli");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int y_part_extent_1 = SIZE1;
    int y_part_extent_0 = SIZE0;
    coli::buffer buff_y_part("buff_y_part", 2, {coli::expr(y_part_extent_1), coli::expr(y_part_extent_0)}, coli::p_uint8, NULL, coli::a_output, &rgbyuv420);
    int u_part_extent_1 = SIZE1;
    int u_part_extent_0 = SIZE0;
    coli::buffer buff_u_part("buff_u_part", 2, {coli::expr(u_part_extent_1), coli::expr(u_part_extent_0)}, coli::p_uint8, NULL, coli::a_output, &rgbyuv420);
    int v_part_extent_1 = SIZE1;
    int v_part_extent_0 = SIZE0;
    coli::buffer buff_v_part("buff_v_part", 2, {coli::expr(v_part_extent_1), coli::expr(v_part_extent_0)}, coli::p_uint8, NULL, coli::a_output, &rgbyuv420);

    // Input buffers.
    int p0_extent_2 = SIZE2;
    int p0_extent_1 = SIZE1;
    int p0_extent_0 = SIZE0;
    coli::buffer buff_p0("buff_p0", 3, {coli::expr(p0_extent_2), coli::expr(p0_extent_1), coli::expr(p0_extent_0)}, coli::p_int16, NULL, coli::a_input, &rgbyuv420);
    coli::computation p0("[p0_extent_2, p0_extent_1, p0_extent_0]->{p0[i2, i1, i0]: (0 <= i2 <= (p0_extent_2 + -1)) and (0 <= i1 <= (p0_extent_1 + -1)) and (0 <= i0 <= (p0_extent_0 + -1))}", expr(), false, coli::p_int16, &rgbyuv420);
    p0.set_access("{p0[i2, i1, i0]->buff_p0[i2, i1, i0]}");


    // Define loop bounds for dimension "y_part_s0_y".
    coli::constant y_part_s0_y_loop_min("y_part_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant y_part_s0_y_loop_extent("y_part_s0_y_loop_extent", coli::expr(y_part_extent_1), coli::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "y_part_s0_x".
    coli::constant y_part_s0_x_loop_min("y_part_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant y_part_s0_x_loop_extent("y_part_s0_x_loop_extent", coli::expr(y_part_extent_0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::computation y_part_s0("[y_part_s0_y_loop_min, y_part_s0_y_loop_extent, y_part_s0_x_loop_min, y_part_s0_x_loop_extent]->{y_part_s0[y_part_s0_y, y_part_s0_x]: "
                        "(y_part_s0_y_loop_min <= y_part_s0_y <= ((y_part_s0_y_loop_min + y_part_s0_y_loop_extent) + -1)) and (y_part_s0_x_loop_min <= y_part_s0_x <= ((y_part_s0_x_loop_min + y_part_s0_x_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_uint8, ((((((coli::expr((int16_t)66)*p0(coli::expr((int32_t)0), coli::idx("y_part_s0_y"), coli::idx("y_part_s0_x"))) + (coli::expr((int16_t)129)*p0(coli::expr((int32_t)1), coli::idx("y_part_s0_y"), coli::idx("y_part_s0_x")))) + (coli::expr((int16_t)25)*p0(coli::expr((int32_t)2), coli::idx("y_part_s0_y"), coli::idx("y_part_s0_x")))) + coli::expr((int16_t)128)) >> coli::expr((int16_t)8)) + coli::expr((int16_t)16))), true, coli::p_uint8, &rgbyuv420);
    y_part_s0.set_access("{y_part_s0[y_part_s0_y, y_part_s0_x]->buff_y_part[y_part_s0_y, y_part_s0_x]}");

    // Define compute level for "y_part".
    y_part_s0.first(computation::root_dimension);

    // Define loop bounds for dimension "u_part_s0_y".
    coli::constant u_part_s0_y_loop_min("u_part_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant u_part_s0_y_loop_extent("u_part_s0_y_loop_extent", coli::expr(u_part_extent_1), coli::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "u_part_s0_x".
    coli::constant u_part_s0_x_loop_min("u_part_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant u_part_s0_x_loop_extent("u_part_s0_x_loop_extent", coli::expr(u_part_extent_0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::computation u_part_s0("[u_part_s0_y_loop_min, u_part_s0_y_loop_extent, u_part_s0_x_loop_min, u_part_s0_x_loop_extent]->{u_part_s0[u_part_s0_y, u_part_s0_x]: "
                        "(u_part_s0_y_loop_min <= u_part_s0_y <= ((u_part_s0_y_loop_min + u_part_s0_y_loop_extent) + -1)) and (u_part_s0_x_loop_min <= u_part_s0_x <= ((u_part_s0_x_loop_min + u_part_s0_x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_uint8, &rgbyuv420);
    coli::constant t0("t0", (coli::idx("u_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    coli::constant t1("t1", (coli::idx("u_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    coli::constant t2("t2", (coli::idx("u_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    coli::constant t3("t3", (coli::idx("u_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    coli::constant t4("t4", (coli::idx("u_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    coli::constant t5("t5", (coli::idx("u_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &u_part_s0, 1, &rgbyuv420);
    u_part_s0.set_expression(coli::expr(coli::o_cast, coli::p_uint8, ((((((coli::expr((int16_t)-38)*p0(coli::expr((int32_t)0), t0(0), t1(0))) - (coli::expr((int16_t)74)*p0(coli::expr((int32_t)1), t2(0), t3(0)))) + (coli::expr((int16_t)112)*p0(coli::expr((int32_t)2), t4(0), t5(0)))) + coli::expr((int16_t)128)) >> coli::expr((int16_t)8)) + coli::expr((int16_t)128))));
    u_part_s0.set_access("{u_part_s0[u_part_s0_y, u_part_s0_x]->buff_u_part[u_part_s0_y, u_part_s0_x]}");

    // Define compute level for "u_part".
    u_part_s0.after(y_part_s0, computation::root_dimension);

    // Define loop bounds for dimension "v_part_s0_y".
    coli::constant v_part_s0_y_loop_min("v_part_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant v_part_s0_y_loop_extent("v_part_s0_y_loop_extent", coli::expr(v_part_extent_1), coli::p_int32, true, NULL, 0, &rgbyuv420);

    // Define loop bounds for dimension "v_part_s0_x".
    coli::constant v_part_s0_x_loop_min("v_part_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::constant v_part_s0_x_loop_extent("v_part_s0_x_loop_extent", coli::expr(v_part_extent_0), coli::p_int32, true, NULL, 0, &rgbyuv420);
    coli::computation v_part_s0("[v_part_s0_y_loop_min, v_part_s0_y_loop_extent, v_part_s0_x_loop_min, v_part_s0_x_loop_extent]->{v_part_s0[v_part_s0_y, v_part_s0_x]: "
                        "(v_part_s0_y_loop_min <= v_part_s0_y <= ((v_part_s0_y_loop_min + v_part_s0_y_loop_extent) + -1)) and (v_part_s0_x_loop_min <= v_part_s0_x <= ((v_part_s0_x_loop_min + v_part_s0_x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_uint8, &rgbyuv420);
    coli::constant t6("t6", (coli::idx("v_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    coli::constant t7("t7", (coli::idx("v_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    coli::constant t8("t8", (coli::idx("v_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    coli::constant t9("t9", (coli::idx("v_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    coli::constant t10("t10", (coli::idx("v_part_s0_y")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    coli::constant t11("t11", (coli::idx("v_part_s0_x")*coli::expr((int32_t)2)), coli::p_int32, false, &v_part_s0, 1, &rgbyuv420);
    v_part_s0.set_expression(coli::expr(coli::o_cast, coli::p_uint8, ((((((coli::expr((int16_t)112)*p0(coli::expr((int32_t)0), t6(0), t7(0))) - (coli::expr((int16_t)94)*p0(coli::expr((int32_t)1), t8(0), t9(0)))) - (coli::expr((int16_t)18)*p0(coli::expr((int32_t)2), t10(0), t11(0)))) + coli::expr((int16_t)128)) >> coli::expr((int16_t)8)) + coli::expr((int16_t)128))));
    v_part_s0.set_access("{v_part_s0[v_part_s0_y, v_part_s0_x]->buff_v_part[v_part_s0_y, v_part_s0_x]}");

    // Define compute level for "v_part".
    v_part_s0.after(u_part_s0, computation::root_dimension);

    // Add schedules.

    rgbyuv420.set_arguments({&buff_p0, &buff_u_part, &buff_v_part, &buff_y_part});
    rgbyuv420.gen_time_processor_domain();
    rgbyuv420.gen_isl_ast();
    rgbyuv420.gen_halide_stmt();
    rgbyuv420.dump_halide_stmt();
    rgbyuv420.gen_halide_obj("build/generated_fct_rgbyuv420.o");

    return 0;
}

