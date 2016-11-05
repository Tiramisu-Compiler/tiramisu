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

    coli::function filter2D_coli("filter2D_coli");
    coli::buffer buff_filter2D("buff_filter2D", 2, {coli::expr(1024), coli::expr(1024)}, coli::p_float32, NULL, coli::a_output, &filter2D_coli);
    coli::buffer buff_b0("buff_b0", 2, {coli::expr(1024), coli::expr(1024)}, coli::p_float32, NULL, coli::a_input, &filter2D_coli);
    coli::computation b0("{b0[i0, i1]: (0 <= i0 <= 1023) and (0 <= i1 <= 1023)}", expr(), false, coli::p_float32, &filter2D_coli);
    b0.set_access("{b0[i0, i1]->buff_b0[i0, i1]}");


    // Define loop bounds for dimension "filter2D_s0_y".
    coli::constant filter2D_s0_y_loop_min("filter2D_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::constant filter2D_s0_y_loop_extent("filter2D_s0_y_loop_extent", coli::expr((int32_t)1024), coli::p_int32, true, NULL, 0, &filter2D_coli);

    // Define loop bounds for dimension "filter2D_s0_x".
    coli::constant filter2D_s0_x_loop_min("filter2D_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::constant filter2D_s0_x_loop_extent("filter2D_s0_x_loop_extent", coli::expr((int32_t)1024), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::computation filter2D_s0("[filter2D_s0_y_loop_min, filter2D_s0_y_loop_extent, filter2D_s0_x_loop_min, filter2D_s0_x_loop_extent]->{filter2D_s0[filter2D_s0_x, filter2D_s0_y]: "
                        "(filter2D_s0_y_loop_min <= filter2D_s0_y <= ((filter2D_s0_y_loop_min + filter2D_s0_y_loop_extent) + -1)) and (filter2D_s0_x_loop_min <= filter2D_s0_x <= ((filter2D_s0_x_loop_min + filter2D_s0_x_loop_extent) + -1))}",
                        ((((((((((((((((((((((((((((((((((((coli::expr((float)0) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)192970))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)4.1051e-41))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)1.08803e+27))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)1.14306e+27))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)0.000710459))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)1.26541e-31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)1.72236e+22))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)0))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)0.000875062))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)7.14284e+31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)7.20648e+31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)-1.08447e-19))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)2979.65))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)0))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)3.07344e+29))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)192970))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)1.5901e+29))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)1.08803e+27))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)0.236757))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)-3.68935e+19))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)1.26903e+31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)1.72236e+22))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)1.26903e+31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)0.000875062))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)0.000710459))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)1.26591e-31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)0.000711031))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)2979.65))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)0))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)3.07344e+29))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-3)))*coli::expr((float)7.20648e+31))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-2)))*coli::expr((float)2.00049))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)-1)))*coli::expr((float)0.000676623))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)0)))*coli::expr((float)0.236757))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)1)))*coli::expr((float)-3.68935e+19))) + (b0(((coli::idx("filter2D_s0_x") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)), ((coli::idx("filter2D_s0_y") + coli::expr((int32_t)3)) + coli::expr((int32_t)2)))*coli::expr((float)1.26903e+31))), true, coli::p_float32, &filter2D_coli);
    filter2D_s0.set_access("{filter2D_s0[filter2D_s0_x, filter2D_s0_y]->buff_filter2D[filter2D_s0_x, filter2D_s0_y]}");

    // Define compute level for "filter2D".
    filter2D_s0.first(computation::root_dimension);

    // Add schedules.
    filter2D_s0.tag_parallel_dimension(1);

    filter2D_coli.set_arguments({&buff_b0, &buff_filter2D});
    filter2D_coli.gen_time_processor_domain();
    filter2D_coli.gen_isl_ast();
    filter2D_coli.gen_halide_stmt();
    filter2D_coli.dump_halide_stmt();
    filter2D_coli.gen_halide_obj("build/generated_fct_filter2D.o");

    return 0;
}
