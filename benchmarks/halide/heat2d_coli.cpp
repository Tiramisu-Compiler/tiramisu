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

    coli::function heat2d_coli("heat2d_coli");

    // Input params.
    float p0 = 0.3;
    float p1 = 0.4;

    Halide::Image<float> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);

    // Output buffers.
    int heat2d_extent_1 = SIZE1;
    int heat2d_extent_0 = SIZE0;
    coli::buffer buff_heat2d("buff_heat2d", 2, {coli::expr(heat2d_extent_1), coli::expr(heat2d_extent_0)}, coli::p_float32, NULL, coli::a_output, &heat2d_coli);

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 2, {coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_float32, NULL, coli::a_input, &heat2d_coli);
    coli::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_float32, &heat2d_coli);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");


    // Define loop bounds for dimension "heat2d_s0_y".
    coli::constant heat2d_s0_y_loop_min("heat2d_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::constant heat2d_s0_y_loop_extent("heat2d_s0_y_loop_extent", coli::expr(heat2d_extent_1), coli::p_int32, true, NULL, 0, &heat2d_coli);

    // Define loop bounds for dimension "heat2d_s0_x".
    coli::constant heat2d_s0_x_loop_min("heat2d_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::constant heat2d_s0_x_loop_extent("heat2d_s0_x_loop_extent", coli::expr(heat2d_extent_0), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::computation heat2d_s0("[heat2d_s0_y_loop_min, heat2d_s0_y_loop_extent, heat2d_s0_x_loop_min, heat2d_s0_x_loop_extent]->{heat2d_s0[heat2d_s0_y, heat2d_s0_x]: "
                        "(heat2d_s0_y_loop_min <= heat2d_s0_y <= ((heat2d_s0_y_loop_min + heat2d_s0_y_loop_extent) + -1)) and (heat2d_s0_x_loop_min <= heat2d_s0_x <= ((heat2d_s0_x_loop_min + heat2d_s0_x_loop_extent) + -1))}",
                        coli::expr((float)0), true, coli::p_float32, &heat2d_coli);
    heat2d_s0.set_access("{heat2d_s0[heat2d_s0_y, heat2d_s0_x]->buff_heat2d[heat2d_s0_y, heat2d_s0_x]}");

    // Define loop bounds for dimension "heat2d_s1_r4__y".
    coli::constant heat2d_s1_r4__y_loop_min("heat2d_s1_r4__y_loop_min", coli::expr((int32_t)1), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::constant heat2d_s1_r4__y_loop_extent("heat2d_s1_r4__y_loop_extent", (coli::expr(input_extent_1) + coli::expr((int32_t)-2)), coli::p_int32, true, NULL, 0, &heat2d_coli);

    // Define loop bounds for dimension "heat2d_s1_r4__x".
    coli::constant heat2d_s1_r4__x_loop_min("heat2d_s1_r4__x_loop_min", coli::expr((int32_t)1), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::constant heat2d_s1_r4__x_loop_extent("heat2d_s1_r4__x_loop_extent", (coli::expr(input_extent_0) + coli::expr((int32_t)-2)), coli::p_int32, true, NULL, 0, &heat2d_coli);
    coli::computation heat2d_s1("[heat2d_s1_r4__y_loop_min, heat2d_s1_r4__y_loop_extent, heat2d_s1_r4__x_loop_min, heat2d_s1_r4__x_loop_extent]->{heat2d_s1[heat2d_s1_r4__y, heat2d_s1_r4__x]: "
                        "(heat2d_s1_r4__y_loop_min <= heat2d_s1_r4__y <= ((heat2d_s1_r4__y_loop_min + heat2d_s1_r4__y_loop_extent) + -1)) and (heat2d_s1_r4__x_loop_min <= heat2d_s1_r4__x <= ((heat2d_s1_r4__x_loop_min + heat2d_s1_r4__x_loop_extent) + -1))}",
                        coli::expr(), true, coli::p_float32, &heat2d_coli);
    heat2d_s1.set_expression(((coli::expr(p0)*input(coli::idx("heat2d_s1_r4__y"), coli::idx("heat2d_s1_r4__x"))) + (coli::expr(p1)*(((input(coli::idx("heat2d_s1_r4__y"), (coli::idx("heat2d_s1_r4__x") + coli::expr((int32_t)1))) + input(coli::idx("heat2d_s1_r4__y"), (coli::idx("heat2d_s1_r4__x") - coli::expr((int32_t)1)))) + input((coli::idx("heat2d_s1_r4__y") + coli::expr((int32_t)1)), coli::idx("heat2d_s1_r4__x"))) + input((coli::idx("heat2d_s1_r4__y") - coli::expr((int32_t)1)), coli::idx("heat2d_s1_r4__x"))))));
    heat2d_s1.set_access("{heat2d_s1[heat2d_s1_r4__y, heat2d_s1_r4__x]->buff_heat2d[heat2d_s1_r4__y, heat2d_s1_r4__x]}");

    // Define compute level for "heat2d".
    heat2d_s0.first(computation::root_dimension);
    heat2d_s1.after(heat2d_s0, computation::root_dimension);

    // Add schedules.
    heat2d_s0.tag_parallel_dimension(0);
    heat2d_s1.tag_parallel_dimension(0);

    heat2d_coli.set_arguments({&buff_input, &buff_heat2d});
    heat2d_coli.gen_time_processor_domain();
    heat2d_coli.gen_isl_ast();
    heat2d_coli.gen_halide_stmt();
    heat2d_coli.dump_halide_stmt();
    heat2d_coli.gen_halide_obj("build/generated_fct_heat2d.o");

    return 0;
}
