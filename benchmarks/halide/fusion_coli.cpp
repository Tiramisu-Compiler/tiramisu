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
    coli::buffer buff_f_s0("buff_f_s0", 2, {coli::expr(100), coli::expr(50)}, coli::p_int32, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_b0("buff_b0", 2, {coli::expr(100), coli::expr(50)}, coli::p_int32, NULL, coli::a_input, &fusion_coli);
    coli::computation b0("{b0[i0, i1]: (0 <= i0 <= 99) and (0 <= i1 <= 49)}", expr(), false, coli::p_int32, &fusion_coli);
    b0.set_access("{b0[i0, i1]->buff_b0[i0, i1]}");


    // Define loop bounds for dimension "f_s0_y".
    coli::constant f_s0_y_loop_min("f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_y_loop_extent("f_s0_y_loop_extent", coli::expr((int32_t)50), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "f_s0_x".
    coli::constant f_s0_x_loop_min("f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_x_loop_extent("f_s0_x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant t0("t0", coli::expr(coli::o_max, coli::idx("f_s0_x"), coli::idx("f_s0_y")), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation f_s0("[f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_x, f_s0_y]: "
                        "(f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_float32, (b0(t0(0), coli::idx("f_s0_y")) >> coli::expr((int32_t)2))), true, coli::p_float32, &fusion_coli);
    f_s0.set_access("{f_s0[f_s0_x, f_s0_y]->buff_f_s0[f_s0_x, f_s0_y]}");

    // Define compute level for "f".
    f_s0.first(computation::root_dimension);

    // Add schedules.

    fusion_coli.set_arguments({&buff_b0, &buff_f_s0});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fusion_coli_test.o");

    return 0;
}