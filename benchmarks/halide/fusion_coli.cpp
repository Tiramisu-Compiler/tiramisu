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

    //int  SIZE0 = 0;
    //int  SIZE1 = 0;

    // COLi generated code here.
    coli::function fusion_coli("fusion_coli");
    coli::buffer buff_f("buff_f", 2, {coli::expr(100), coli::expr(100)}, coli::p_int32, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_b0("buff_b0", 2, {coli::expr(100), coli::expr(100)}, coli::p_int32, NULL, coli::a_input, &fusion_coli);
    coli::computation b0("{b0[i0, i1]: (0 <= i0 <= 99) and (0 <= i1 <= 99)}", expr(), false, coli::p_int32, &fusion_coli);
    b0.set_access("{b0[i0, i1]->buff_b0[i0, i1]}");
    coli::constant _f_s0_y_loop_min("_f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _f_s0_y_loop_extent("_f_s0_y_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _f_s0_x_loop_min("_f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _f_s0_x_loop_extent("_f_s0_x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation f("[_f_s0_y_loop_min, _f_s0_y_loop_extent, _f_s0_x_loop_min, _f_s0_x_loop_extent]->{f[_f_s0_x, _f_s0_y]: (_f_s0_y_loop_min <= _f_s0_y <= ((_f_s0_y_loop_min + _f_s0_y_loop_extent) + -1)) and (_f_s0_x_loop_min <= _f_s0_x <= ((_f_s0_x_loop_min + _f_s0_x_loop_extent) + -1))}", b0(coli::idx("_f_s0_x"), coli::idx("_f_s0_y")), true, coli::p_int32, &fusion_coli);
    f.set_access("{f[_f_s0_x, _f_s0_y]->buff_f[_f_s0_x, _f_s0_y]}");
    fusion_coli.set_arguments({&buff_f, &buff_b0});


    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion.o");

    // Some debugging
    fusion_coli.dump_iteration_domain();
    fusion_coli.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    fusion_coli.dump(true);

    return 0;
}
