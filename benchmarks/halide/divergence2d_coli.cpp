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

    coli::function divergence2d_coli("divergence2d_coli");

    // Input params.
    float p0 = 0.3;
    float p1 = 0.4;

    Halide::Image<float> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);

    // Output buffers.
    int divergence2d_extent_1 = SIZE1;
    int divergence2d_extent_0 = SIZE0;
    coli::buffer buff_divergence2d("buff_divergence2d", 2, {coli::expr(divergence2d_extent_1), coli::expr(divergence2d_extent_0)}, coli::p_float32, NULL, coli::a_output, &divergence2d_coli);

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 2, {coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_float32, NULL, coli::a_input, &divergence2d_coli);
    coli::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_float32, &divergence2d_coli);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "divergence2d_s0_y".
    coli::constant divergence2d_s0_y_loop_min("divergence2d_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::constant divergence2d_s0_y_loop_extent("divergence2d_s0_y_loop_extent", coli::expr(divergence2d_extent_1), coli::p_int32, true, NULL, 0, &divergence2d_coli);

    // Define loop bounds for dimension "divergence2d_s0_x".
    coli::constant divergence2d_s0_x_loop_min("divergence2d_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::constant divergence2d_s0_x_loop_extent("divergence2d_s0_x_loop_extent", coli::expr(divergence2d_extent_0), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::computation divergence2d_s0("[divergence2d_s0_y_loop_min, divergence2d_s0_y_loop_extent, divergence2d_s0_x_loop_min, divergence2d_s0_x_loop_extent]->{divergence2d_s0[divergence2d_s0_y, divergence2d_s0_x]: "
                        "(divergence2d_s0_y_loop_min <= divergence2d_s0_y <= ((divergence2d_s0_y_loop_min + divergence2d_s0_y_loop_extent) + -1)) and (divergence2d_s0_x_loop_min <= divergence2d_s0_x <= ((divergence2d_s0_x_loop_min + divergence2d_s0_x_loop_extent) + -1))}",
                        coli::expr((float)0), true, coli::p_float32, &divergence2d_coli);
    divergence2d_s0.set_access("{divergence2d_s0[divergence2d_s0_y, divergence2d_s0_x]->buff_divergence2d[divergence2d_s0_y, divergence2d_s0_x]}");

    // Define loop bounds for dimension "divergence2d_s1_r4__y".
    coli::constant divergence2d_s1_r4__y_loop_min("divergence2d_s1_r4__y_loop_min", coli::expr((int32_t)1), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::constant divergence2d_s1_r4__y_loop_extent("divergence2d_s1_r4__y_loop_extent", (coli::expr(input_extent_1) + coli::expr((int32_t)-2)), coli::p_int32, true, NULL, 0, &divergence2d_coli);

    // Define loop bounds for dimension "divergence2d_s1_r4__x".
    coli::constant divergence2d_s1_r4__x_loop_min("divergence2d_s1_r4__x_loop_min", coli::expr((int32_t)1), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::constant divergence2d_s1_r4__x_loop_extent("divergence2d_s1_r4__x_loop_extent", (coli::expr(input_extent_0) + coli::expr((int32_t)-2)), coli::p_int32, true, NULL, 0, &divergence2d_coli);
    coli::computation divergence2d_s1("[divergence2d_s1_r4__y_loop_min, divergence2d_s1_r4__y_loop_extent, divergence2d_s1_r4__x_loop_min, divergence2d_s1_r4__x_loop_extent]->{divergence2d_s1[divergence2d_s1_r4__y, divergence2d_s1_r4__x]: "
                        "(divergence2d_s1_r4__y_loop_min <= divergence2d_s1_r4__y <= ((divergence2d_s1_r4__y_loop_min + divergence2d_s1_r4__y_loop_extent) + -1)) and (divergence2d_s1_r4__x_loop_min <= divergence2d_s1_r4__x <= ((divergence2d_s1_r4__x_loop_min + divergence2d_s1_r4__x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_float32, &divergence2d_coli);
    divergence2d_s1.set_expression(((coli::expr(p0)*(input(coli::idx("divergence2d_s1_r4__y"), (coli::idx("divergence2d_s1_r4__x") + coli::expr((int32_t)1))) + input(coli::idx("divergence2d_s1_r4__y"), (coli::idx("divergence2d_s1_r4__x") - coli::expr((int32_t)1))))) + (coli::expr(p1)*(input((coli::idx("divergence2d_s1_r4__y") + coli::expr((int32_t)1)), coli::idx("divergence2d_s1_r4__x")) + input((coli::idx("divergence2d_s1_r4__y") - coli::expr((int32_t)1)), coli::idx("divergence2d_s1_r4__x"))))));
    divergence2d_s1.set_access("{divergence2d_s1[divergence2d_s1_r4__y, divergence2d_s1_r4__x]->buff_divergence2d[divergence2d_s1_r4__y, divergence2d_s1_r4__x]}");

    // Define compute level for "divergence2d".
    divergence2d_s0.first(computation::root_dimension);
    divergence2d_s1.after(divergence2d_s0, computation::root_dimension);

    // Add schedules.
    divergence2d_s0.tag_parallel_dimension(0);
    divergence2d_s1.tag_parallel_dimension(0);

    divergence2d_coli.set_arguments({&buff_input, &buff_divergence2d});
    divergence2d_coli.gen_time_processor_domain();
    divergence2d_coli.gen_isl_ast();
    divergence2d_coli.gen_halide_stmt();
    divergence2d_coli.dump_halide_stmt();
    divergence2d_coli.gen_halide_obj("build/generated_fct_divergence2d.o");

    return 0;
}

