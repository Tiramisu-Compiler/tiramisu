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
    coli::buffer buff_affine_s0("buff_affine_s0", 2, {coli::expr(256), coli::expr(128)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_b0("buff_b0", 2, {coli::expr(256), coli::expr(128)}, coli::p_uint8, NULL, coli::a_input, &fusion_coli);
    coli::computation b0("{b0[i0, i1]: (0 <= i0 <= 255) and (0 <= i1 <= 127)}", expr(), false, coli::p_uint8, &fusion_coli);
    b0.set_access("{b0[i0, i1]->buff_b0[i0, i1]}");


    // Define loop bounds for dimension "affine_s0_y".
    coli::constant affine_s0_y_loop_min("affine_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant affine_s0_y_loop_extent("affine_s0_y_loop_extent", coli::expr((int32_t)128), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "affine_s0_x".
    coli::constant affine_s0_x_loop_min("affine_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant affine_s0_x_loop_extent("affine_s0_x_loop_extent", coli::expr((int32_t)256), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation affine_s0("[affine_s0_y_loop_min, affine_s0_y_loop_extent, affine_s0_x_loop_min, affine_s0_x_loop_extent]->{affine_s0[affine_s0_x, affine_s0_y]: "
                        "(affine_s0_y_loop_min <= affine_s0_y <= ((affine_s0_y_loop_min + affine_s0_y_loop_extent) + -1)) and (affine_s0_x_loop_min <= affine_s0_x <= ((affine_s0_x_loop_min + affine_s0_x_loop_extent) + -1))}",
                        ((((coli::expr(coli::o_cast, coli::p_float32, b0(coli::expr(coli::o_max, coli::expr(coli::o_min, coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))), coli::expr((int32_t)128)), coli::expr((int32_t)0)), coli::expr(coli::o_max, coli::expr(coli::o_min, coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))), coli::expr((int32_t)256)), coli::expr((int32_t)0))))*(coli::expr((float)1.000000) - ((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))))) + (coli::expr(coli::o_cast, coli::p_float32, b0(coli::expr(coli::o_max, coli::expr(coli::o_min, (coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))) + coli::expr((int32_t)1)), coli::expr((int32_t)128)), coli::expr((int32_t)0)), coli::expr(coli::o_max, coli::expr(coli::o_min, coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))), coli::expr((int32_t)256)), coli::expr((int32_t)0))))*((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000))))))*(coli::expr((float)1.000000) - ((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))))) + (((coli::expr(coli::o_cast, coli::p_float32, b0(coli::expr(coli::o_max, coli::expr(coli::o_min, coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))), coli::expr((int32_t)128)), coli::expr((int32_t)0)), coli::expr(coli::o_max, coli::expr(coli::o_min, (coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))) + coli::expr((int32_t)1)), coli::expr((int32_t)256)), coli::expr((int32_t)0))))*(coli::expr((float)1.000000) - ((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))))) + (coli::expr(coli::o_cast, coli::p_float32, b0(coli::expr(coli::o_max, coli::expr(coli::o_min, (coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))) + coli::expr((int32_t)1)), coli::expr((int32_t)128)), coli::expr((int32_t)0)), coli::expr(coli::o_max, coli::expr(coli::o_min, (coli::expr(coli::o_cast, coli::p_int32, coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))) + coli::expr((int32_t)1)), coli::expr((int32_t)256)), coli::expr((int32_t)0))))*((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000))))))*((((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)) - coli::expr(o_floor, (((coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_y"))) + (coli::expr((float)0.100000)*coli::expr(coli::o_cast, coli::p_float32, coli::idx("affine_s0_x")))) + coli::expr((float)0.100000)))))), true, coli::p_float32, &fusion_coli);
    affine_s0.set_access("{affine_s0[affine_s0_x, affine_s0_y]->buff_affine_s0[affine_s0_x, affine_s0_y]}");

    // Define compute level for "affine".
    affine_s0.first(computation::root_dimension);

    // Add schedules.
    affine_s0.tag_parallel_dimension(1);

    fusion_coli.set_arguments({&buff_b0, &buff_affine_s0});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fusion_coli_test.o");

    return 0;
}

