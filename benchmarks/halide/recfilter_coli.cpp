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

    coli::function recfilter_coli("recfilter_coli");

    // Input params.
    float a0 = 0.3;
    float a1 = 0.4;
    float a2 = 0.3;

    // Output buffers.
    int rec_filter_extent_1 = 1024;
    int rec_filter_extent_0 = 1024;
    coli::buffer buff_rec_filter("buff_rec_filter", 2, {coli::expr(rec_filter_extent_1), coli::expr(rec_filter_extent_0)}, coli::p_float32, NULL, coli::a_output, &recfilter_coli);

    // Input buffers.
    int input_extent_1 = 1024;
    int input_extent_0 = 1024;
    coli::buffer buff_input("buff_input", 2, {coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_float32, NULL, coli::a_input, &recfilter_coli);
    coli::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_float32, &recfilter_coli);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "rec_filter_s0_v3".
    coli::constant rec_filter_s0_v3_loop_min("rec_filter_s0_v3_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s0_v3_loop_extent("rec_filter_s0_v3_loop_extent", coli::expr(coli::o_max, coli::expr(rec_filter_extent_1), (coli::expr(input_extent_1) + coli::expr((int32_t)-1))), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s0_v2".
    coli::constant rec_filter_s0_v2_loop_min("rec_filter_s0_v2_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s0_v2_loop_extent("rec_filter_s0_v2_loop_extent", coli::expr(coli::o_max, coli::expr(rec_filter_extent_0), (coli::expr(input_extent_0) + coli::expr((int32_t)-1))), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::computation rec_filter_s0("[rec_filter_s0_v3_loop_min, rec_filter_s0_v3_loop_extent, rec_filter_s0_v2_loop_min, rec_filter_s0_v2_loop_extent]->{rec_filter_s0[rec_filter_s0_v3, rec_filter_s0_v2]: "
                        "(rec_filter_s0_v3_loop_min <= rec_filter_s0_v3 <= ((rec_filter_s0_v3_loop_min + rec_filter_s0_v3_loop_extent) + -1)) and (rec_filter_s0_v2_loop_min <= rec_filter_s0_v2 <= ((rec_filter_s0_v2_loop_min + rec_filter_s0_v2_loop_extent) + -1))}",
                        input(coli::idx("rec_filter_s0_v3"), coli::idx("rec_filter_s0_v2")), true, coli::p_float32, &recfilter_coli);
    rec_filter_s0.set_access("{rec_filter_s0[rec_filter_s0_v3, rec_filter_s0_v2]->buff_rec_filter[rec_filter_s0_v3, rec_filter_s0_v2]}");

    // Define loop bounds for dimension "rec_filter_s1_r4__y".
    coli::constant rec_filter_s1_r4__y_loop_min("rec_filter_s1_r4__y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s1_r4__y_loop_extent("rec_filter_s1_r4__y_loop_extent", (coli::expr(input_extent_1) + coli::expr((int32_t)-1)), coli::p_int32, true, NULL, 0, &recfilter_coli);

    // Define loop bounds for dimension "rec_filter_s1_r4__x".
    coli::constant rec_filter_s1_r4__x_loop_min("rec_filter_s1_r4__x_loop_min", coli::expr((int32_t)2), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::constant rec_filter_s1_r4__x_loop_extent("rec_filter_s1_r4__x_loop_extent", (coli::expr(input_extent_0) + coli::expr((int32_t)-3)), coli::p_int32, true, NULL, 0, &recfilter_coli);
    coli::computation rec_filter_s1("[rec_filter_s1_r4__y_loop_min, rec_filter_s1_r4__y_loop_extent, rec_filter_s1_r4__x_loop_min, rec_filter_s1_r4__x_loop_extent]->{rec_filter_s1[rec_filter_s1_r4__y, rec_filter_s1_r4__x]: "
                        "(rec_filter_s1_r4__y_loop_min <= rec_filter_s1_r4__y <= ((rec_filter_s1_r4__y_loop_min + rec_filter_s1_r4__y_loop_extent) + -1)) and (rec_filter_s1_r4__x_loop_min <= rec_filter_s1_r4__x <= ((rec_filter_s1_r4__x_loop_min + rec_filter_s1_r4__x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_float32, &recfilter_coli);
    rec_filter_s1.set_expression((((coli::expr(a0)*rec_filter_s0(coli::idx("rec_filter_s1_r4__y"), coli::idx("rec_filter_s1_r4__x"))) + (coli::expr(a1)*rec_filter_s0(coli::idx("rec_filter_s1_r4__y"), (coli::idx("rec_filter_s1_r4__x") - coli::expr((int32_t)1))))) + (coli::expr(a2)*rec_filter_s0(coli::idx("rec_filter_s1_r4__y"), (coli::idx("rec_filter_s1_r4__x") - coli::expr((int32_t)2))))));
    rec_filter_s1.set_access("{rec_filter_s1[rec_filter_s1_r4__y, rec_filter_s1_r4__x]->buff_rec_filter[rec_filter_s1_r4__y, rec_filter_s1_r4__x]}");

    // Define compute level for "rec_filter".
    rec_filter_s0.first(computation::root_dimension);
    rec_filter_s1.after(rec_filter_s0, computation::root_dimension);

    // Add schedules.
    rec_filter_s0.tag_parallel_dimension(0);

    recfilter_coli.set_arguments({&buff_input, &buff_rec_filter});
    recfilter_coli.gen_time_processor_domain();
    recfilter_coli.gen_isl_ast();
    recfilter_coli.gen_halide_stmt();
    recfilter_coli.dump_halide_stmt();
    recfilter_coli.gen_halide_obj("build/generated_fct_recfilter.o");

    return 0;
}

