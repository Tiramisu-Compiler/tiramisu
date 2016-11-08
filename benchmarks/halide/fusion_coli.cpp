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

    coli::function fusion_coli("fusion_coli");

    // Output buffers.
    int RGB2Gray_extent_1 = 512;
    int RGB2Gray_extent_0 = 1024;
    coli::buffer buff_RGB2Gray("buff_RGB2Gray", 2, {coli::expr(RGB2Gray_extent_1), coli::expr(RGB2Gray_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);

    // Input buffers.
    int b0_extent_2 = 3;
    int b0_extent_1 = 512;
    int b0_extent_0 = 1024;
    coli::buffer buff_b0("buff_b0", 3, {coli::expr(b0_extent_2), coli::expr(b0_extent_1), coli::expr(b0_extent_0)}, coli::p_uint8, NULL, coli::a_input, &fusion_coli);
    coli::computation b0("[b0_extent_2, b0_extent_1, b0_extent_0]->{b0[i2, i1, i0]: (0 <= i2 <= (b0_extent_2 + -1)) and (0 <= i1 <= (b0_extent_1 + -1)) and (0 <= i0 <= (b0_extent_0 + -1))}", expr(), false, coli::p_uint8, &fusion_coli);
    b0.set_access("{b0[i2, i1, i0]->buff_b0[i2, i1, i0]}");


    // Define loop bounds for dimension "RGB2Gray_s0_v1".
    coli::constant RGB2Gray_s0_v1_loop_min("RGB2Gray_s0_v1_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant RGB2Gray_s0_v1_loop_extent("RGB2Gray_s0_v1_loop_extent", coli::expr(RGB2Gray_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "RGB2Gray_s0_v0".
    coli::constant RGB2Gray_s0_v0_loop_min("RGB2Gray_s0_v0_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant RGB2Gray_s0_v0_loop_extent("RGB2Gray_s0_v0_loop_extent", coli::expr(RGB2Gray_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation RGB2Gray_s0("[RGB2Gray_s0_v1_loop_min, RGB2Gray_s0_v1_loop_extent, RGB2Gray_s0_v0_loop_min, RGB2Gray_s0_v0_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_v1, RGB2Gray_s0_v0]: "
                        "(RGB2Gray_s0_v1_loop_min <= RGB2Gray_s0_v1 <= ((RGB2Gray_s0_v1_loop_min + RGB2Gray_s0_v1_loop_extent) + -1)) and (RGB2Gray_s0_v0_loop_min <= RGB2Gray_s0_v0 <= ((RGB2Gray_s0_v0_loop_min + RGB2Gray_s0_v0_loop_extent) + -1))}",
                        ((b0(coli::expr((int32_t)2), coli::idx("RGB2Gray_s0_v1"), coli::idx("RGB2Gray_s0_v0")) + b0(coli::expr((int32_t)1), coli::idx("RGB2Gray_s0_v1"), coli::idx("RGB2Gray_s0_v0"))) + b0(coli::expr((int32_t)0), coli::idx("RGB2Gray_s0_v1"), coli::idx("RGB2Gray_s0_v0"))), true, coli::p_uint8, &fusion_coli);
    RGB2Gray_s0.set_access("{RGB2Gray_s0[RGB2Gray_s0_v1, RGB2Gray_s0_v0]->buff_RGB2Gray[RGB2Gray_s0_v1, RGB2Gray_s0_v0]}");

    // Define compute level for "RGB2Gray".
    RGB2Gray_s0.first(computation::root_dimension);

    // Add schedules.
    RGB2Gray_s0.tag_parallel_dimension(0);

    fusion_coli.set_arguments({&buff_b0, &buff_RGB2Gray});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion_coli_test.o");

    return 0;
}

