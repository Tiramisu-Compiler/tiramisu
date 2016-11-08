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
    int C_extent_1 = 1024;
    int C_extent_0 = 1024;
    coli::buffer buff_C("buff_C", 2, {coli::expr(C_extent_1), coli::expr(C_extent_0)}, coli::p_int8, NULL, coli::a_output, &fusion_coli);

    // Input buffers.
    int A_extent_1 = 1024;
    int A_extent_0 = 1024;
    coli::buffer buff_A("buff_A", 2, {coli::expr(A_extent_1), coli::expr(A_extent_0)}, coli::p_int8, NULL, coli::a_input, &fusion_coli);
    coli::computation A("[A_extent_1, A_extent_0]->{A[i1, i0]: (0 <= i1 <= (A_extent_1 + -1)) and (0 <= i0 <= (A_extent_0 + -1))}", expr(), false, coli::p_int8, &fusion_coli);
    A.set_access("{A[i1, i0]->buff_A[i1, i0]}");

    int B_extent_1 = 1024;
    int B_extent_0 = 1024;
    coli::buffer buff_B("buff_B", 2, {coli::expr(B_extent_1), coli::expr(B_extent_0)}, coli::p_int8, NULL, coli::a_input, &fusion_coli);
    coli::computation B("[B_extent_1, B_extent_0]->{B[i1, i0]: (0 <= i1 <= (B_extent_1 + -1)) and (0 <= i0 <= (B_extent_0 + -1))}", expr(), false, coli::p_int8, &fusion_coli);
    B.set_access("{B[i1, i0]->buff_B[i1, i0]}");


    // Define loop bounds for dimension "C_s0_y".
    coli::constant C_s0_y_loop_min("C_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant C_s0_y_loop_extent("C_s0_y_loop_extent", coli::expr(C_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "C_s0_x".
    coli::constant C_s0_x_loop_min("C_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant C_s0_x_loop_extent("C_s0_x_loop_extent", coli::expr(C_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation C_s0("[C_s0_y_loop_min, C_s0_y_loop_extent, C_s0_x_loop_min, C_s0_x_loop_extent]->{C_s0[C_s0_y, C_s0_x]: "
                        "(C_s0_y_loop_min <= C_s0_y <= ((C_s0_y_loop_min + C_s0_y_loop_extent) + -1)) and (C_s0_x_loop_min <= C_s0_x <= ((C_s0_x_loop_min + C_s0_x_loop_extent) + -1))}",
                        coli::expr((int8_t)0), true, coli::p_int8, &fusion_coli);
    C_s0.set_access("{C_s0[C_s0_y, C_s0_x]->buff_C[C_s0_y, C_s0_x]}");

    // Define loop bounds for dimension "C_s1_y".
    coli::constant C_s1_y_loop_min("C_s1_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant C_s1_y_loop_extent("C_s1_y_loop_extent", coli::expr(C_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "C_s1_x".
    coli::constant C_s1_x_loop_min("C_s1_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant C_s1_x_loop_extent("C_s1_x_loop_extent", coli::expr(C_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "C_s1_r4__x".
    coli::constant C_s1_r4__x_loop_min("C_s1_r4__x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant C_s1_r4__x_loop_extent("C_s1_r4__x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation C_s1("[C_s1_y_loop_min, C_s1_y_loop_extent, C_s1_x_loop_min, C_s1_x_loop_extent, C_s1_r4__x_loop_min, C_s1_r4__x_loop_extent]->{C_s1[C_s1_y, C_s1_x, C_s1_r4__x]: "
                        "(C_s1_y_loop_min <= C_s1_y <= ((C_s1_y_loop_min + C_s1_y_loop_extent) + -1)) and (C_s1_x_loop_min <= C_s1_x <= ((C_s1_x_loop_min + C_s1_x_loop_extent) + -1)) and (C_s1_r4__x_loop_min <= C_s1_r4__x <= ((C_s1_r4__x_loop_min + C_s1_r4__x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_int8, &fusion_coli);
    C_s1.set_expression((C_s1(coli::idx("C_s1_y"), coli::idx("C_s1_x"), (coli::idx("C_s1_r4__x") - coli::expr((int32_t)1))) + (A(coli::idx("C_s1_r4__x"), coli::idx("C_s1_x"))*B(coli::idx("C_s1_y"), coli::idx("C_s1_r4__x")))));
    C_s1.set_access("{C_s1[C_s1_y, C_s1_x, C_s1_r4__x]->buff_C[C_s1_y, C_s1_x]}");

    // Define compute level for "C".
    C_s0.first(computation::root_dimension);
    C_s1.after(C_s0, computation::root_dimension);

    // Add schedules.

    fusion_coli.set_arguments({&buff_A, &buff_B, &buff_C});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion_coli_test.o");

    return 0;
}

