#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    tiramisu::function gaussian_tiramisu("gaussian_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int gaussian_extent_2 = SIZE2;
    int gaussian_extent_1 = SIZE1 - 8;
    int gaussian_extent_0 = SIZE0 - 8;
    tiramisu::buffer buff_gaussian("buff_gaussian", 3, {tiramisu::expr(gaussian_extent_2), tiramisu::expr(gaussian_extent_1), tiramisu::expr(gaussian_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &gaussian_tiramisu);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", 3, {tiramisu::expr(input_extent_2), tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_input, &gaussian_tiramisu);
    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &gaussian_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");

    int kernelx_extent_0 = 5;
    tiramisu::buffer buff_kernelx("buff_kernelx", 1, {tiramisu::expr(kernelx_extent_0)}, tiramisu::p_float32, NULL, tiramisu::a_input, &gaussian_tiramisu);
    tiramisu::computation kernelx("[kernelx_extent_0]->{kernelx[i0]: (0 <= i0 <= (kernelx_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &gaussian_tiramisu);
    kernelx.set_access("{kernelx[i0]->buff_kernelx[i0]}");

    int kernely_extent_0 = 5;
    tiramisu::buffer buff_kernely("buff_kernely", 1, {tiramisu::expr(kernely_extent_0)}, tiramisu::p_float32, NULL, tiramisu::a_input, &gaussian_tiramisu);
    tiramisu::computation kernely("[kernely_extent_0]->{kernely[i0]: (0 <= i0 <= (kernely_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &gaussian_tiramisu);
    kernely.set_access("{kernely[i0]->buff_kernely[i0]}");


    // Define temporary buffers for "gaussian_x".
    tiramisu::buffer buff_gaussian_x("buff_gaussian_x", 3, {tiramisu::expr(gaussian_extent_2), tiramisu::expr(gaussian_extent_1 + 4), tiramisu::expr(gaussian_extent_0)}, tiramisu::p_float32, NULL, tiramisu::a_temporary, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_x_s0_c".
    tiramisu::constant gaussian_x_s0_c_loop_min("gaussian_x_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_c_loop_extent("gaussian_x_s0_c_loop_extent", tiramisu::expr(gaussian_extent_2), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_x_s0_y".
    tiramisu::constant gaussian_x_s0_y_loop_min("gaussian_x_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_y_loop_extent("gaussian_x_s0_y_loop_extent", (tiramisu::expr(gaussian_extent_1) + tiramisu::expr((int32_t)4)), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_x_s0_x".
    tiramisu::constant gaussian_x_s0_x_loop_min("gaussian_x_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_x_loop_extent("gaussian_x_s0_x_loop_extent", tiramisu::expr(gaussian_extent_0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::computation gaussian_x_s0("[gaussian_x_s0_c_loop_min, gaussian_x_s0_c_loop_extent, gaussian_x_s0_y_loop_min, gaussian_x_s0_y_loop_extent, gaussian_x_s0_x_loop_min, gaussian_x_s0_x_loop_extent]->{gaussian_x_s0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]: "
                        "(gaussian_x_s0_c_loop_min <= gaussian_x_s0_c <= ((gaussian_x_s0_c_loop_min + gaussian_x_s0_c_loop_extent) + -1)) and (gaussian_x_s0_y_loop_min <= gaussian_x_s0_y <= ((gaussian_x_s0_y_loop_min + gaussian_x_s0_y_loop_extent) + -1)) and (gaussian_x_s0_x_loop_min <= gaussian_x_s0_x <= ((gaussian_x_s0_x_loop_min + gaussian_x_s0_x_loop_extent) + -1))}",
                        (((((tiramisu::expr((float)0) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("gaussian_x_s0_c"), tiramisu::idx("gaussian_x_s0_y"), (tiramisu::idx("gaussian_x_s0_x") + tiramisu::expr((int32_t)0))))*kernelx(tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("gaussian_x_s0_c"), tiramisu::idx("gaussian_x_s0_y"), (tiramisu::idx("gaussian_x_s0_x") + tiramisu::expr((int32_t)1))))*kernelx(tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("gaussian_x_s0_c"), tiramisu::idx("gaussian_x_s0_y"), (tiramisu::idx("gaussian_x_s0_x") + tiramisu::expr((int32_t)2))))*kernelx(tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("gaussian_x_s0_c"), tiramisu::idx("gaussian_x_s0_y"), (tiramisu::idx("gaussian_x_s0_x") + tiramisu::expr((int32_t)3))))*kernelx(tiramisu::expr((int32_t)3)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::idx("gaussian_x_s0_c"), tiramisu::idx("gaussian_x_s0_y"), (tiramisu::idx("gaussian_x_s0_x") + tiramisu::expr((int32_t)4))))*kernelx(tiramisu::expr((int32_t)4)))), true, tiramisu::p_float32, &gaussian_tiramisu);
    gaussian_x_s0.set_access("{gaussian_x_s0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]}");


    // Define loop bounds for dimension "gaussian_s0_c".
    tiramisu::constant gaussian_s0_c_loop_min("gaussian_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_c_loop_extent("gaussian_s0_c_loop_extent", tiramisu::expr(gaussian_extent_2), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_s0_y".
    tiramisu::constant gaussian_s0_y_loop_min("gaussian_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_y_loop_extent("gaussian_s0_y_loop_extent", tiramisu::expr(gaussian_extent_1), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_s0_x".
    tiramisu::constant gaussian_s0_x_loop_min("gaussian_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_x_loop_extent("gaussian_s0_x_loop_extent", tiramisu::expr(gaussian_extent_0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::computation gaussian_s0("[gaussian_s0_c_loop_min, gaussian_s0_c_loop_extent, gaussian_s0_y_loop_min, gaussian_s0_y_loop_extent, gaussian_s0_x_loop_min, gaussian_s0_x_loop_extent]->{gaussian_s0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]: "
                        "(gaussian_s0_c_loop_min <= gaussian_s0_c <= ((gaussian_s0_c_loop_min + gaussian_s0_c_loop_extent) + -1)) and (gaussian_s0_y_loop_min <= gaussian_s0_y <= ((gaussian_s0_y_loop_min + gaussian_s0_y_loop_extent) + -1)) and (gaussian_s0_x_loop_min <= gaussian_s0_x <= ((gaussian_s0_x_loop_min + gaussian_s0_x_loop_extent) + -1))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint8, (((((tiramisu::expr((float)0) + (gaussian_x_s0(tiramisu::idx("gaussian_s0_c"), (tiramisu::idx("gaussian_s0_y") + tiramisu::expr((int32_t)0)), tiramisu::idx("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)0)))) + (gaussian_x_s0(tiramisu::idx("gaussian_s0_c"), (tiramisu::idx("gaussian_s0_y") + tiramisu::expr((int32_t)1)), tiramisu::idx("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)1)))) + (gaussian_x_s0(tiramisu::idx("gaussian_s0_c"), (tiramisu::idx("gaussian_s0_y") + tiramisu::expr((int32_t)2)), tiramisu::idx("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)2)))) + (gaussian_x_s0(tiramisu::idx("gaussian_s0_c"), (tiramisu::idx("gaussian_s0_y") + tiramisu::expr((int32_t)3)), tiramisu::idx("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)3)))) + (gaussian_x_s0(tiramisu::idx("gaussian_s0_c"), (tiramisu::idx("gaussian_s0_y") + tiramisu::expr((int32_t)4)), tiramisu::idx("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)4))))), true, tiramisu::p_uint8, &gaussian_tiramisu);
    gaussian_s0.set_access("{gaussian_s0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]->buff_gaussian[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]}");


    // Define compute level for "gaussian_x".
    gaussian_x_s0.first(computation::root_dimension);
    // Define compute level for "gaussian".
    gaussian_s0.after(gaussian_x_s0, computation::root_dimension);

    // Add schedules.

    gaussian_tiramisu.set_arguments({&buff_input, &buff_kernelx, &buff_kernely, &buff_gaussian});
    gaussian_tiramisu.gen_time_processor_domain();
    gaussian_tiramisu.gen_isl_ast();
    gaussian_tiramisu.gen_halide_stmt();
    gaussian_tiramisu.dump_halide_stmt();
    gaussian_tiramisu.gen_halide_obj("build/generated_fct_gaussian.o");

    return 0;
}
