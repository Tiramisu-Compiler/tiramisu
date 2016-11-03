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
    coli::buffer buff_f("buff_f", 2, {coli::expr(100), coli::expr(50)}, coli::p_int32, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_in("buff_in", 2, {coli::expr((int32_t)100), coli::expr((int32_t)50)}, coli::p_int32, NULL, coli::a_temporary, &fusion_coli);

    // Define loop bounds for dimension "_in_s0_x".
    coli::constant _in_s0_x_loop_min("_in_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _in_s0_x_loop_extent("_in_s0_x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "_in_s0_y".
    coli::constant _in_s0_y_loop_min("_in_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _in_s0_y_loop_extent("_in_s0_y_loop_extent", coli::expr((int32_t)50), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation in("[_in_s0_x_loop_min, _in_s0_x_loop_extent, _in_s0_y_loop_min, _in_s0_y_loop_extent]->{in[_in_s0_x, _in_s0_y]: "
                        "(_in_s0_x_loop_min <= _in_s0_x <= ((_in_s0_x_loop_min + _in_s0_x_loop_extent) + -1)) and (_in_s0_y_loop_min <= _in_s0_y <= ((_in_s0_y_loop_min + _in_s0_y_loop_extent) + -1))}",
                        coli::expr((int32_t)13), true, coli::p_int32, &fusion_coli);
    in.set_access("{in[_in_s0_x, _in_s0_y]->buff_in[_in_s0_x, _in_s0_y]}");

    // Define loop bounds for dimension "_f_s0_y".
    coli::constant _f_s0_y_loop_min("_f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _f_s0_y_loop_extent("_f_s0_y_loop_extent", coli::expr((int32_t)50), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "_f_s0_x".
    coli::constant _f_s0_x_loop_min("_f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant _f_s0_x_loop_extent("_f_s0_x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation f("[_f_s0_y_loop_min, _f_s0_y_loop_extent, _f_s0_x_loop_min, _f_s0_x_loop_extent]->{f[_f_s0_x, _f_s0_y]: "
                        "(_f_s0_y_loop_min <= _f_s0_y <= ((_f_s0_y_loop_min + _f_s0_y_loop_extent) + -1)) and (_f_s0_x_loop_min <= _f_s0_x <= ((_f_s0_x_loop_min + _f_s0_x_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_float32, (in(coli::idx("_f_s0_x"), coli::idx("_f_s0_y")) >> coli::expr((int32_t)2))), true, coli::p_float32, &fusion_coli);
    f.set_access("{f[_f_s0_x, _f_s0_y]->buff_f[_f_s0_x, _f_s0_y]}");

    // Add schedules.
    in.first(computation::root_dimension);
    in.tag_parallel_dimension(1);
    f.after(in, computation::root_dimension);
    f.tag_vector_dimension(1);

    fusion_coli.set_arguments({&buff_f});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fusion_coli_test.o");

    return 0;
}
