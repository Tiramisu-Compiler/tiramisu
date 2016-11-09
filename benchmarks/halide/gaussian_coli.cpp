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

    coli::function gaussian_coli("gaussian_coli");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int gaussian_extent_2 = SIZE2;
    int gaussian_extent_1 = SIZE1 - 8;
    int gaussian_extent_0 = SIZE0 - 8;
    coli::buffer buff_gaussian("buff_gaussian", 3, {coli::expr(gaussian_extent_2), coli::expr(gaussian_extent_1), coli::expr(gaussian_extent_0)}, coli::p_uint8, NULL, coli::a_output, &gaussian_coli);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 3, {coli::expr(input_extent_2), coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_uint8, NULL, coli::a_input, &gaussian_coli);
    coli::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_uint8, &gaussian_coli);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");

    int kernelx_extent_0 = 5;
    coli::buffer buff_kernelx("buff_kernelx", 1, {coli::expr(kernelx_extent_0)}, coli::p_float32, NULL, coli::a_input, &gaussian_coli);
    coli::computation kernelx("[kernelx_extent_0]->{kernelx[i0]: (0 <= i0 <= (kernelx_extent_0 + -1))}", expr(), false, coli::p_float32, &gaussian_coli);
    kernelx.set_access("{kernelx[i0]->buff_kernelx[i0]}");

    int kernely_extent_0 = 5;
    coli::buffer buff_kernely("buff_kernely", 1, {coli::expr(kernely_extent_0)}, coli::p_float32, NULL, coli::a_input, &gaussian_coli);
    coli::computation kernely("[kernely_extent_0]->{kernely[i0]: (0 <= i0 <= (kernely_extent_0 + -1))}", expr(), false, coli::p_float32, &gaussian_coli);
    kernely.set_access("{kernely[i0]->buff_kernely[i0]}");


    // Define temporary buffers for "gaussian_x".
    coli::buffer buff_gaussian_x("buff_gaussian_x", 3, {coli::expr(gaussian_extent_2), coli::expr(gaussian_extent_1 + 4), coli::expr(gaussian_extent_0)}, coli::p_float32, NULL, coli::a_temporary, &gaussian_coli);

    // Define loop bounds for dimension "gaussian_x_s0_c".
    coli::constant gaussian_x_s0_c_loop_min("gaussian_x_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_x_s0_c_loop_extent("gaussian_x_s0_c_loop_extent", coli::expr(gaussian_extent_2), coli::p_int32, true, NULL, 0, &gaussian_coli);

    // Define loop bounds for dimension "gaussian_x_s0_y".
    coli::constant gaussian_x_s0_y_loop_min("gaussian_x_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_x_s0_y_loop_extent("gaussian_x_s0_y_loop_extent", (coli::expr(gaussian_extent_1) + coli::expr((int32_t)4)), coli::p_int32, true, NULL, 0, &gaussian_coli);

    // Define loop bounds for dimension "gaussian_x_s0_x".
    coli::constant gaussian_x_s0_x_loop_min("gaussian_x_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_x_s0_x_loop_extent("gaussian_x_s0_x_loop_extent", coli::expr(gaussian_extent_0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::computation gaussian_x_s0("[gaussian_x_s0_c_loop_min, gaussian_x_s0_c_loop_extent, gaussian_x_s0_y_loop_min, gaussian_x_s0_y_loop_extent, gaussian_x_s0_x_loop_min, gaussian_x_s0_x_loop_extent]->{gaussian_x_s0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]: "
                        "(gaussian_x_s0_c_loop_min <= gaussian_x_s0_c <= ((gaussian_x_s0_c_loop_min + gaussian_x_s0_c_loop_extent) + -1)) and (gaussian_x_s0_y_loop_min <= gaussian_x_s0_y <= ((gaussian_x_s0_y_loop_min + gaussian_x_s0_y_loop_extent) + -1)) and (gaussian_x_s0_x_loop_min <= gaussian_x_s0_x <= ((gaussian_x_s0_x_loop_min + gaussian_x_s0_x_loop_extent) + -1))}",
                        (((((coli::expr((float)0) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("gaussian_x_s0_c"), coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)0))))*kernelx(coli::expr((int32_t)0)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("gaussian_x_s0_c"), coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)1))))*kernelx(coli::expr((int32_t)1)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("gaussian_x_s0_c"), coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)2))))*kernelx(coli::expr((int32_t)2)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("gaussian_x_s0_c"), coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)3))))*kernelx(coli::expr((int32_t)3)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("gaussian_x_s0_c"), coli::idx("gaussian_x_s0_y"), (coli::idx("gaussian_x_s0_x") + coli::expr((int32_t)4))))*kernelx(coli::expr((int32_t)4)))), true, coli::p_float32, &gaussian_coli);
    gaussian_x_s0.set_access("{gaussian_x_s0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]}");


    // Define loop bounds for dimension "gaussian_s0_c".
    coli::constant gaussian_s0_c_loop_min("gaussian_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_s0_c_loop_extent("gaussian_s0_c_loop_extent", coli::expr(gaussian_extent_2), coli::p_int32, true, NULL, 0, &gaussian_coli);

    // Define loop bounds for dimension "gaussian_s0_y".
    coli::constant gaussian_s0_y_loop_min("gaussian_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_s0_y_loop_extent("gaussian_s0_y_loop_extent", coli::expr(gaussian_extent_1), coli::p_int32, true, NULL, 0, &gaussian_coli);

    // Define loop bounds for dimension "gaussian_s0_x".
    coli::constant gaussian_s0_x_loop_min("gaussian_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::constant gaussian_s0_x_loop_extent("gaussian_s0_x_loop_extent", coli::expr(gaussian_extent_0), coli::p_int32, true, NULL, 0, &gaussian_coli);
    coli::computation gaussian_s0("[gaussian_s0_c_loop_min, gaussian_s0_c_loop_extent, gaussian_s0_y_loop_min, gaussian_s0_y_loop_extent, gaussian_s0_x_loop_min, gaussian_s0_x_loop_extent]->{gaussian_s0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]: "
                        "(gaussian_s0_c_loop_min <= gaussian_s0_c <= ((gaussian_s0_c_loop_min + gaussian_s0_c_loop_extent) + -1)) and (gaussian_s0_y_loop_min <= gaussian_s0_y <= ((gaussian_s0_y_loop_min + gaussian_s0_y_loop_extent) + -1)) and (gaussian_s0_x_loop_min <= gaussian_s0_x <= ((gaussian_s0_x_loop_min + gaussian_s0_x_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_uint8, (((((coli::expr((float)0) + (gaussian_x_s0(coli::idx("gaussian_s0_c"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)0)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)0)))) + (gaussian_x_s0(coli::idx("gaussian_s0_c"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)1)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)1)))) + (gaussian_x_s0(coli::idx("gaussian_s0_c"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)2)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)2)))) + (gaussian_x_s0(coli::idx("gaussian_s0_c"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)3)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)3)))) + (gaussian_x_s0(coli::idx("gaussian_s0_c"), (coli::idx("gaussian_s0_y") + coli::expr((int32_t)4)), coli::idx("gaussian_s0_x"))*kernely(coli::expr((int32_t)4))))), true, coli::p_uint8, &gaussian_coli);
    gaussian_s0.set_access("{gaussian_s0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]->buff_gaussian[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]}");


    // Define compute level for "gaussian_x".
    gaussian_x_s0.first(computation::root_dimension);
    // Define compute level for "gaussian".
    gaussian_s0.after(gaussian_x_s0, computation::root_dimension);

    // Add schedules.

    gaussian_coli.set_arguments({&buff_input, &buff_kernelx, &buff_kernely, &buff_gaussian});
    gaussian_coli.gen_time_processor_domain();
    gaussian_coli.gen_isl_ast();
    gaussian_coli.gen_halide_stmt();
    gaussian_coli.dump_halide_stmt();
    gaussian_coli.gen_halide_obj("build/generated_fct_gaussian_coli.o");

    return 0;
}
