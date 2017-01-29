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


using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    coli::function recfilter_coli("recfilter_coli");

    // Input params.
    float a0 = 0.7;
    float a1 = 0.2;
    float a2 = 0.1;

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int rec_filter_extent_2 = SIZE2;
    int rec_filter_extent_1 = SIZE1;
    int rec_filter_extent_0 = SIZE0;
    coli::buffer buff_rec_filter("buff_rec_filter", 3, {coli::expr(rec_filter_extent_2), coli::expr(rec_filter_extent_1), coli::expr(rec_filter_extent_0)}, coli::p_uint8, NULL, coli::a_output, &recfilter_coli);

    // Input buffers.
    int b0_extent_2 = SIZE2;
    int b0_extent_1 = SIZE1;
    int b0_extent_0 = SIZE0;
    coli::buffer buff_b0("buff_b0", 3, {coli::expr(b0_extent_2), coli::expr(b0_extent_1), coli::expr(b0_extent_0)}, coli::p_uint8, NULL, coli::a_input, &recfilter_coli);
    coli::computation b0("[b0_extent_2, b0_extent_1, b0_extent_0]->{b0[i2, i1, i0]: (0 <= i2 <= (b0_extent_2 + -1)) and (0 <= i1 <= (b0_extent_1 + -1)) and (0 <= i0 <= (b0_extent_0 + -1))}", expr(), false, coli::p_uint8, &recfilter_coli);
    b0.set_access("{b0[i2, i1, i0]->buff_b0[i2, i1, i0]}");


    // Define loop bounds for dimension "rec_filter_s0_c".
    coli::constant rec_filter_s0_c_loop_min("rec_filter_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s0_c_loop_extent("rec_filter_s0_c_loop_extent", coli::expr(rec_filter_extent_2), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s0_y".
    coli::constant rec_filter_s0_y_loop_min("rec_filter_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s0_y_loop_extent("rec_filter_s0_y_loop_extent", coli::expr(coli::o_max, coli::expr(rec_filter_extent_1), coli::expr((int32_t)3537)), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s0_x".
    coli::constant rec_filter_s0_x_loop_min("rec_filter_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s0_x_loop_extent("rec_filter_s0_x_loop_extent", coli::expr(coli::o_max, coli::expr(rec_filter_extent_0), coli::expr((int32_t)2115)), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::computation rec_filter_s0("[rec_filter_s0_c_loop_min, rec_filter_s0_c_loop_extent, rec_filter_s0_y_loop_min, rec_filter_s0_y_loop_extent, rec_filter_s0_x_loop_min, rec_filter_s0_x_loop_extent]->{rec_filter_s0[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]: "
                        "(rec_filter_s0_c_loop_min <= rec_filter_s0_c <= ((rec_filter_s0_c_loop_min + rec_filter_s0_c_loop_extent) + -1)) and (rec_filter_s0_y_loop_min <= rec_filter_s0_y <= ((rec_filter_s0_y_loop_min + rec_filter_s0_y_loop_extent) + -1)) and (rec_filter_s0_x_loop_min <= rec_filter_s0_x <= ((rec_filter_s0_x_loop_min + rec_filter_s0_x_loop_extent) + -1))}",
                        b0(coli::idx("rec_filter_s0_c"), coli::idx("rec_filter_s0_y"), coli::idx("rec_filter_s0_x")), true, coli::p_uint8, &recfilter_coli);
    rec_filter_s0.set_access("{rec_filter_s0[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]->buff_rec_filter[rec_filter_s0_c, rec_filter_s0_y, rec_filter_s0_x]}");

    // Define loop bounds for dimension "rec_filter_s1_c".
    coli::constant rec_filter_s1_c_loop_min("rec_filter_s1_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s1_c_loop_extent("rec_filter_s1_c_loop_extent", coli::expr(rec_filter_extent_2), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s1_r4__y".
    coli::constant rec_filter_s1_r4__y_loop_min("rec_filter_s1_r4__y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s1_r4__y_loop_extent("rec_filter_s1_r4__y_loop_extent", coli::expr((int32_t)3537), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s1_r4__x".
    coli::constant rec_filter_s1_r4__x_loop_min("rec_filter_s1_r4__x_loop_min", coli::expr((int32_t)2), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s1_r4__x_loop_extent("rec_filter_s1_r4__x_loop_extent", coli::expr((int32_t)2113), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::computation rec_filter_s1("[rec_filter_s1_c_loop_min, rec_filter_s1_c_loop_extent, rec_filter_s1_r4__y_loop_min, rec_filter_s1_r4__y_loop_extent, rec_filter_s1_r4__x_loop_min, rec_filter_s1_r4__x_loop_extent]->{rec_filter_s1[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]: "
                        "(rec_filter_s1_c_loop_min <= rec_filter_s1_c <= ((rec_filter_s1_c_loop_min + rec_filter_s1_c_loop_extent) + -1)) and (rec_filter_s1_r4__y_loop_min <= rec_filter_s1_r4__y <= ((rec_filter_s1_r4__y_loop_min + rec_filter_s1_r4__y_loop_extent) + -1)) and (rec_filter_s1_r4__x_loop_min <= rec_filter_s1_r4__x <= ((rec_filter_s1_r4__x_loop_min + rec_filter_s1_r4__x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_uint8, &recfilter_coli);
    rec_filter_s1.set_expression(coli::expr(coli::o_cast, coli::p_uint8, (((coli::expr(a0)*coli::expr(coli::o_cast, coli::p_float32, rec_filter_s0(coli::idx("rec_filter_s1_c"), coli::idx("rec_filter_s1_r4__y"), coli::idx("rec_filter_s1_r4__x")))) + (coli::expr(a1)*coli::expr(coli::o_cast, coli::p_float32, rec_filter_s0(coli::idx("rec_filter_s1_c"), coli::idx("rec_filter_s1_r4__y"), (coli::idx("rec_filter_s1_r4__x") - coli::expr((int32_t)1)))))) + (coli::expr(a2)*coli::expr(coli::o_cast, coli::p_float32, rec_filter_s0(coli::idx("rec_filter_s1_c"), coli::idx("rec_filter_s1_r4__y"), (coli::idx("rec_filter_s1_r4__x") - coli::expr((int32_t)2))))))));
    rec_filter_s1.set_access("{rec_filter_s1[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]->buff_rec_filter[rec_filter_s1_c, rec_filter_s1_r4__y, rec_filter_s1_r4__x]}");

    // Define compute level for "rec_filter".
    rec_filter_s0.first(computation::root_dimension);
    rec_filter_s1.after(rec_filter_s0, computation::root_dimension);

    // Add schedules.
    rec_filter_s0.tag_parallel_dimension(1);
    rec_filter_s0.tag_parallel_dimension(0);
    rec_filter_s1.tag_parallel_dimension(1);
    rec_filter_s1.tag_parallel_dimension(0);

    recfilter_coli.set_arguments({&buff_b0, &buff_rec_filter});
    recfilter_coli.gen_time_processor_domain();
    recfilter_coli.gen_isl_ast();
    recfilter_coli.gen_halide_stmt();
    recfilter_coli.dump_halide_stmt();
    recfilter_coli.gen_halide_obj("build/generated_fct_recfilter.o");

    return 0;
}
