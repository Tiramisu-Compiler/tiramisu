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

    coli::function gaussian3x3_coli("gaussian3x3_coli");

    // Output buffers.
    int gaussian_extent_1 = 1024;
    int gaussian_extent_0 = 1024;
    coli::buffer buff_gaussian("buff_gaussian", 2, {coli::expr(gaussian_extent_1), coli::expr(gaussian_extent_0)}, coli::p_float32, NULL, coli::a_output, &gaussian3x3_coli);

    // Input buffers.
    int input_extent_1 = 1024;
    int input_extent_0 = 1024;
    coli::buffer buff_input("buff_input", 2, {coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_float32, NULL, coli::a_input, &gaussian3x3_coli);
    coli::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_float32, &gaussian3x3_coli);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");

    int kernelx_extent_0 = 1024;
    coli::buffer buff_kernelx("buff_kernelx", 1, {coli::expr(kernelx_extent_0)}, coli::p_float32, NULL, coli::a_input, &gaussian3x3_coli);
    coli::computation kernelx("[kernelx_extent_0]->{kernelx[i0]: (0 <= i0 <= (kernelx_extent_0 + -1))}", expr(), false, coli::p_float32, &gaussian3x3_coli);
    kernelx.set_access("{kernelx[i0]->buff_kernelx[i0]}");

    int kernely_extent_0 = 1024;
    coli::buffer buff_kernely("buff_kernely", 1, {coli::expr(kernely_extent_0)}, coli::p_float32, NULL, coli::a_input, &gaussian3x3_coli);
    coli::computation kernely("[kernely_extent_0]->{kernely[i0]: (0 <= i0 <= (kernely_extent_0 + -1))}", expr(), false, coli::p_float32, &gaussian3x3_coli);
    kernely.set_access("{kernely[i0]->buff_kernely[i0]}");


    // Define temporary buffers for "gaussian_x".
    coli::buffer buff_gaussian_x("buff_gaussian_x", 2, {(coli::expr(gaussian_extent_1) + coli::expr((int32_t)6)), coli::expr(gaussian_extent_0)}, coli::p_float32, NULL, coli::a_temporary, &gaussian3x3_coli);

    // Define loop bounds for dimension "gaussian_x_s0_y".
    coli::constant gaussian_x_s0_y_loop_min("gaussian_x_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::constant gaussian_x_s0_y_loop_extent("gaussian_x_s0_y_loop_extent", (coli::expr(gaussian_extent_1) + coli::expr((int32_t)6)), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);

    // Define loop bounds for dimension "gaussian_x_s0_x".
    coli::constant gaussian_x_s0_x_loop_min("gaussian_x_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::constant gaussian_x_s0_x_loop_extent("gaussian_x_s0_x_loop_extent", coli::expr(gaussian_extent_0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::computation gaussian_x_s0("[gaussian_x_s0_y_loop_min, gaussian_x_s0_y_loop_extent, gaussian_x_s0_x_loop_min, gaussian_x_s0_x_loop_extent]->{gaussian_x_s0[gaussian_x_s0_y, gaussian_x_s0_x]: "
                        "(gaussian_x_s0_y_loop_min <= gaussian_x_s0_y <= ((gaussian_x_s0_y_loop_min + gaussian_x_s0_y_loop_extent) + -1)) and (gaussian_x_s0_x_loop_min <= gaussian_x_s0_x <= ((gaussian_x_s0_x_loop_min + gaussian_x_s0_x_loop_extent) + -1))}",
                        (((((((coli::expr((float)0) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)0)))*kernelx(coli::expr((int32_t)0)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)1)))*kernelx(coli::expr((int32_t)1)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)2)))*kernelx(coli::expr((int32_t)2)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)3)))*kernelx(coli::expr((int32_t)3)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)4)))*kernelx(coli::expr((int32_t)4)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)5)))*kernelx(coli::expr((int32_t)5)))) + (input(coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)6)))*kernelx(coli::expr((int32_t)6)))), true, coli::p_float32, &gaussian3x3_coli);
    gaussian_x_s0.set_access("{gaussian_x_s0[gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x[gaussian_x_s0_y, gaussian_x_s0_x]}");

    // Define compute level for "gaussian_x".
    gaussian_x_s0.first(computation::root_dimension);

    // Define loop bounds for dimension "gaussian_s0_y".
    coli::constant gaussian_s0_y_loop_min("gaussian_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::constant gaussian_s0_y_loop_extent("gaussian_s0_y_loop_extent", coli::expr(gaussian_extent_1), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);

    // Define loop bounds for dimension "gaussian_s0_x".
    coli::constant gaussian_s0_x_loop_min("gaussian_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::constant gaussian_s0_x_loop_extent("gaussian_s0_x_loop_extent", coli::expr(gaussian_extent_0), coli::p_int32, true, NULL, 0, &gaussian3x3_coli);
    coli::computation gaussian_s0("[gaussian_s0_y_loop_min, gaussian_s0_y_loop_extent, gaussian_s0_x_loop_min, gaussian_s0_x_loop_extent]->{gaussian_s0[gaussian_s0_y, gaussian_s0_x]: "
                        "(gaussian_s0_y_loop_min <= gaussian_s0_y <= ((gaussian_s0_y_loop_min + gaussian_s0_y_loop_extent) + -1)) and (gaussian_s0_x_loop_min <= gaussian_s0_x <= ((gaussian_s0_x_loop_min + gaussian_s0_x_loop_extent) + -1))}",
                        (((((((coli::expr((float)0) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)0)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)0)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)1)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)1)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)2)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)2)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)3)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)3)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)4)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)4)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)5)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)5)))) + (gaussian_x_s0((coli::idx("gaussian_s0_y") + coli::expr((int32_t)6)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)6)))), true, coli::p_float32, &gaussian3x3_coli);
    gaussian_s0.set_access("{gaussian_s0[gaussian_s0_y, gaussian_s0_x]->buff_gaussian[gaussian_s0_y, gaussian_s0_x]}");

    // Define compute level for "gaussian".
    gaussian_s0.after(gaussian_x_s0, computation::root_dimension);

    // Add schedules.

    gaussian3x3_coli.set_arguments({&buff_input, &buff_kernelx, &buff_kernely, &buff_gaussian});
    gaussian3x3_coli.gen_time_processor_domain();
    gaussian3x3_coli.gen_isl_ast();
    gaussian3x3_coli.gen_halide_stmt();
    gaussian3x3_coli.dump_halide_stmt();
    gaussian3x3_coli.gen_halide_obj("build/generated_fct_gaussian3x3.o");

    return 0;
}


