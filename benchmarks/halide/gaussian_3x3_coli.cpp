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

    coli::function gaussian_3x3_coli("gaussian_3x3_coli");
    coli::buffer buff_gaussian("buff_gaussian", 2, {coli::expr(1024), coli::expr(1024)}, coli::p_float32, NULL, coli::a_output, &gaussian_3x3_coli);
    coli::buffer buff_b0("buff_b0", 2, {coli::expr(1024), coli::expr(1024)}, coli::p_float32, NULL, coli::a_input, &gaussian_3x3_coli);
    coli::computation b0("{b0[i0, i1]: (0 <= i0 <= 1023) and (0 <= i1 <= 1023)}", expr(), false, coli::p_float32, &gaussian_3x3_coli);
    b0.set_access("{b0[i0, i1]->buff_b0[i0, i1]}");


    // Define temporary buffers for "gaussian_x".
    coli::buffer buff_gaussian_x("buff_gaussian_x", 2, {coli::expr((int32_t)1024), coli::expr((int32_t)1030)}, coli::p_float32, NULL, coli::a_temporary, &gaussian_3x3_coli);

    // Define loop bounds for dimension "gaussian_x_s0_y".
    coli::constant gaussian_x_s0_y_loop_min("gaussian_x_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::constant gaussian_x_s0_y_loop_extent("gaussian_x_s0_y_loop_extent", coli::expr((int32_t)1030), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);

    // Define loop bounds for dimension "gaussian_x_s0_x".
    coli::constant gaussian_x_s0_x_loop_min("gaussian_x_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::constant gaussian_x_s0_x_loop_extent("gaussian_x_s0_x_loop_extent", coli::expr((int32_t)1024), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::computation gaussian_x_s0("[gaussian_x_s0_y_loop_min, gaussian_x_s0_y_loop_extent, gaussian_x_s0_x_loop_min, gaussian_x_s0_x_loop_extent]->{gaussian_x_s0[gaussian_x_s0_x, gaussian_x_s0_y]: "
                        "(gaussian_x_s0_y_loop_min <= gaussian_x_s0_y <= ((gaussian_x_s0_y_loop_min + gaussian_x_s0_y_loop_extent) + -1)) and (gaussian_x_s0_x_loop_min <= gaussian_x_s0_x <= ((gaussian_x_s0_x_loop_min + gaussian_x_s0_x_loop_extent) + -1))}",
                        (((((((coli::expr((float)0) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)0)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)2.19557e-32))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)1)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)1.4013e-45))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)2)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)0))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)3)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)7.34684e-40))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)4)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)1.65045e-41))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)5)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)0))) + (b0((coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)6)), coli::idx("gaussian_x_s0_y"))*coli::expr((float)0))), true, coli::p_float32, &gaussian_3x3_coli);
    gaussian_x_s0.set_access("{gaussian_x_s0[gaussian_x_s0_x, gaussian_x_s0_y]->buff_gaussian_x[gaussian_x_s0_x, gaussian_x_s0_y]}");

    // Define compute level for "gaussian_x".
    gaussian_x_s0.first(computation::root_dimension);

    // Define loop bounds for dimension "gaussian_s0_y".
    coli::constant gaussian_s0_y_loop_min("gaussian_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::constant gaussian_s0_y_loop_extent("gaussian_s0_y_loop_extent", coli::expr((int32_t)1024), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);

    // Define loop bounds for dimension "gaussian_s0_x".
    coli::constant gaussian_s0_x_loop_min("gaussian_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::constant gaussian_s0_x_loop_extent("gaussian_s0_x_loop_extent", coli::expr((int32_t)1024), coli::p_int32, true, NULL, 0, &gaussian_3x3_coli);
    coli::computation gaussian_s0("[gaussian_s0_y_loop_min, gaussian_s0_y_loop_extent, gaussian_s0_x_loop_min, gaussian_s0_x_loop_extent]->{gaussian_s0[gaussian_s0_x, gaussian_s0_y]: "
                        "(gaussian_s0_y_loop_min <= gaussian_s0_y <= ((gaussian_s0_y_loop_min + gaussian_s0_y_loop_extent) + -1)) and (gaussian_s0_x_loop_min <= gaussian_s0_x <= ((gaussian_s0_x_loop_min + gaussian_s0_x_loop_extent) + -1))}",
                        (((((((coli::expr((float)0) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)0)))*coli::expr((float)7.42408e-42))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)1)))*coli::expr((float)1.4013e-45))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)2)))*coli::expr((float)1.30321e-42))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)3)))*coli::expr((float)2.8026e-45))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)4)))*coli::expr((float)7.42968e-42))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)5)))*coli::expr((float)1.4013e-45))) + (gaussian_x_s0(coli::idx("gaussian_s0_x"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)6)))*coli::expr((float)1.4013e-44))), true, coli::p_float32, &gaussian_3x3_coli);
    gaussian_s0.set_access("{gaussian_s0[gaussian_s0_x, gaussian_s0_y]->buff_gaussian[gaussian_s0_x, gaussian_s0_y]}");

    // Define compute level for "gaussian".
    gaussian_s0.after(gaussian_x_s0, computation::root_dimension);

    // Add schedules.

    gaussian_3x3_coli.set_arguments({&buff_b0, &buff_gaussian});
    gaussian_3x3_coli.gen_time_processor_domain();
    gaussian_3x3_coli.gen_isl_ast();
    gaussian_3x3_coli.gen_halide_stmt();
    gaussian_3x3_coli.dump_halide_stmt();
    gaussian_3x3_coli.gen_halide_obj("build/generated_gaussian_3x3_coli_test.o");

    return 0;
}

