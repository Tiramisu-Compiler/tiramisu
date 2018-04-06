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

    tiramisu::function recfilter_tiramisu("recfilter_tiramisu");

    // Input params.
    float a0 = 0.7;
    float a1 = 0.2;
    float a2 = 0.1;

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int rec_filter_extent_2 = SIZE2;
    int rec_filter_extent_1 = SIZE1;
    int rec_filter_extent_0 = SIZE0;
    tiramisu::buffer buff_rec_filter("buff_rec_filter", {tiramisu::expr(rec_filter_extent_2), tiramisu::expr(rec_filter_extent_1), tiramisu::expr(rec_filter_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &recfilter_tiramisu);

    // Input buffers.
    int b0_extent_2 = SIZE2;
    int b0_extent_1 = SIZE1;
    int b0_extent_0 = SIZE0;
    tiramisu::buffer buff_b0("buff_b0", {tiramisu::expr(b0_extent_2), tiramisu::expr(b0_extent_1), tiramisu::expr(b0_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &recfilter_tiramisu);
    tiramisu::computation b0("[b0_extent_2, b0_extent_1, b0_extent_0]->{b0[i2, i1, i0]: (0 <= i2 <= (b0_extent_2 + -1)) and (0 <= i1 <= (b0_extent_1 + -1)) and (0 <= i0 <= (b0_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &recfilter_tiramisu);
    b0.set_access("{b0[i2, i1, i0]->buff_b0[i2, i1, i0]}");


    // Define loop bounds for dimension "rec_filter_s0_c".
    tiramisu::constant rec_filter_s0_c_loop_min("rec_filter_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s0_c_loop_extent("rec_filter_s0_c_loop_extent", tiramisu::expr(rec_filter_extent_2), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);

    // Define loop bounds for dimension "rec_filter_s0_y".
    tiramisu::constant rec_filter_s0_y_loop_min("rec_filter_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s0_y_loop_extent("rec_filter_s0_y_loop_extent", tiramisu::expr(tiramisu::o_max, tiramisu::expr(rec_filter_extent_1), tiramisu::expr((int32_t)3537)), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);

    // Define loop bounds for dimension "rec_filter_s0_x".
    tiramisu::constant rec_filter_s0_x_loop_min("rec_filter_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s0_x_loop_extent("rec_filter_s0_x_loop_extent", tiramisu::expr(tiramisu::o_max, tiramisu::expr(rec_filter_extent_0), tiramisu::expr((int32_t)2115)), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::computation rec_filter_s0("[rec_filter_s0_c_loop_min, rec_filter_s0_c_loop_extent, rec_filter_s0_y_loop_min, rec_filter_s0_y_loop_extent, rec_filter_s0_x_loop_min, rec_filter_s0_x_loop_extent]->{rec_filter_s0[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]: "
                        "(rec_filter_s0_c_loop_min <= rec_filter_s0_c <= ((rec_filter_s0_c_loop_min + rec_filter_s0_c_loop_extent) + -1)) and (rec_filter_s0_y_loop_min <= rec_filter_s0_y <= ((rec_filter_s0_y_loop_min + rec_filter_s0_y_loop_extent) + -1)) and (rec_filter_s0_x_loop_min <= rec_filter_s0_x <= ((rec_filter_s0_x_loop_min + rec_filter_s0_x_loop_extent) + -1))}",
                        b0(tiramisu::var("rec_filter_s0_c"), tiramisu::var("rec_filter_s0_y"), tiramisu::var("rec_filter_s0_x")), true, tiramisu::p_uint8, &recfilter_tiramisu);
    rec_filter_s0.set_access("{rec_filter_s0[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]->buff_rec_filter[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]}");

    // Define loop bounds for dimension "rec_filter_s1_c".
    tiramisu::constant rec_filter_s1_c_loop_min("rec_filter_s1_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s1_c_loop_extent("rec_filter_s1_c_loop_extent", tiramisu::expr(rec_filter_extent_2), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);

    // Define loop bounds for dimension "rec_filter_s1_r4__y".
    tiramisu::constant rec_filter_s1_r4__y_loop_min("rec_filter_s1_r4__y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s1_r4__y_loop_extent("rec_filter_s1_r4__y_loop_extent", tiramisu::expr((int32_t)3537), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);

    // Define loop bounds for dimension "rec_filter_s1_r4__x".
    tiramisu::constant rec_filter_s1_r4__x_loop_min("rec_filter_s1_r4__x_loop_min", tiramisu::expr((int32_t)2), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::constant rec_filter_s1_r4__x_loop_extent("rec_filter_s1_r4__x_loop_extent", tiramisu::expr((int32_t)2113), tiramisu::p_int32, true, NULL, 0, &recfilter_tiramisu);
    tiramisu::computation rec_filter_s1("[rec_filter_s1_c_loop_min, rec_filter_s1_c_loop_extent, rec_filter_s1_r4__y_loop_min, rec_filter_s1_r4__y_loop_extent, rec_filter_s1_r4__x_loop_min, rec_filter_s1_r4__x_loop_extent]->{rec_filter_s1[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]: "
                        "(rec_filter_s1_c_loop_min <= rec_filter_s1_c <= ((rec_filter_s1_c_loop_min + rec_filter_s1_c_loop_extent) + -1)) and (rec_filter_s1_r4__y_loop_min <= rec_filter_s1_r4__y <= ((rec_filter_s1_r4__y_loop_min + rec_filter_s1_r4__y_loop_extent) + -1)) and (rec_filter_s1_r4__x_loop_min <= rec_filter_s1_r4__x <= ((rec_filter_s1_r4__x_loop_min + rec_filter_s1_r4__x_loop_extent) + -1))}",
                        tiramisu::expr(), true, tiramisu::p_uint8, &recfilter_tiramisu);
    rec_filter_s1.set_expression(tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((tiramisu::expr(a0)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s0(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y"), tiramisu::var("rec_filter_s1_r4__x")))) + (tiramisu::expr(a1)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s0(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y"), (tiramisu::var("rec_filter_s1_r4__x") - tiramisu::expr((int32_t)1)))))) + (tiramisu::expr(a2)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s0(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y"), (tiramisu::var("rec_filter_s1_r4__x") - tiramisu::expr((int32_t)2))))))));
    rec_filter_s1.set_access("{rec_filter_s1[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]->buff_rec_filter[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]}");


    tiramisu::computation rec_filter_s2("[rec_filter_s1_c_loop_min, rec_filter_s1_c_loop_extent, rec_filter_s1_r4__y_loop_min, rec_filter_s1_r4__y_loop_extent, rec_filter_s1_r4__x_loop_min, rec_filter_s1_r4__x_loop_extent]->{rec_filter_s2[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]: "
                        "(rec_filter_s1_c_loop_min <= rec_filter_s1_c <= ((rec_filter_s1_c_loop_min + rec_filter_s1_c_loop_extent) + -1)) and (rec_filter_s1_r4__y_loop_min <= rec_filter_s1_r4__y <= ((rec_filter_s1_r4__y_loop_min + rec_filter_s1_r4__y_loop_extent) + -1)) and (rec_filter_s1_r4__x_loop_min <= rec_filter_s1_r4__x <= ((rec_filter_s1_r4__x_loop_min + rec_filter_s1_r4__x_loop_extent) + -1))}",
                        tiramisu::expr(), true, tiramisu::p_uint8, &recfilter_tiramisu);
    rec_filter_s2.set_expression(tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((tiramisu::expr(a0)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s1(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y"), tiramisu::var("rec_filter_s1_r4__x")))) + (tiramisu::expr(a1)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s1(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y")  - tiramisu::expr((int32_t)1), (tiramisu::var("rec_filter_s1_r4__x")))))) + (tiramisu::expr(a2)*tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, rec_filter_s1(tiramisu::var("rec_filter_s1_c"), tiramisu::var("rec_filter_s1_r4__y") - tiramisu::expr((int32_t)2), (tiramisu::var("rec_filter_s1_r4__x"))))))));
    rec_filter_s2.set_access("{rec_filter_s2[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]->buff_rec_filter[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]}");

    // Define compute level for "rec_filter".
    rec_filter_s1.after(rec_filter_s0, computation::root);
    rec_filter_s2.after(rec_filter_s1, computation::root);

    // Add schedules.
    rec_filter_s0.tag_parallel_level(tiramisu::var("rec_filter_s0_y"));
    rec_filter_s0.tag_parallel_level(tiramisu::var("rec_filter_s0_c"));
    rec_filter_s0.vectorize(tiramisu::var("rec_filter_s0_c"), 8);
    rec_filter_s1.tag_parallel_level(tiramisu::var("rec_filter_s1_c"));
    rec_filter_s1.tag_parallel_level(tiramisu::var("rec_filter_s1_r4__y"));
    rec_filter_s2.tag_parallel_level(tiramisu::var("rec_filter_s1_c"));
    rec_filter_s2.tag_parallel_level(tiramisu::var("rec_filter_s1_r4__y"));

    recfilter_tiramisu.set_arguments({&buff_b0, &buff_rec_filter});
    recfilter_tiramisu.gen_time_space_domain();
    recfilter_tiramisu.gen_isl_ast();
    recfilter_tiramisu.gen_halide_stmt();
    recfilter_tiramisu.dump_halide_stmt();
    recfilter_tiramisu.gen_halide_obj("build/generated_fct_recfilter.o");

    return 0;
}
