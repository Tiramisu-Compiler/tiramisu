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

using namespace tiramisu;

void generate_function_1(std::string name, int size0, int size1, int val0, int val1)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    tiramisu::function test_reduction_operator("test_reduction_operator");

    // Output buffers.
    int f_extent_1 = size1;
    int f_extent_0 = size0;
    tiramisu::buffer buff_f("buff_f", 2, {tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &test_reduction_operator);

    // Input buffers.

    // Define temporary buffers for "input".
    tiramisu::buffer buff_input("buff_input", 3, {tiramisu::expr((int32_t)10), tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_temporary, &test_reduction_operator);

    // Define loop bounds for dimension "input_s0_z".
    tiramisu::constant input_s0_z_loop_min("input_s0_z_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant input_s0_z_loop_extent("input_s0_z_loop_extent", tiramisu::expr((int32_t)10), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);

    // Define loop bounds for dimension "input_s0_y".
    tiramisu::constant input_s0_y_loop_min("input_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant input_s0_y_loop_extent("input_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);

    // Define loop bounds for dimension "input_s0_x".
    tiramisu::constant input_s0_x_loop_min("input_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant input_s0_x_loop_extent("input_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::computation input_s0("[input_s0_z_loop_min, input_s0_z_loop_extent, input_s0_y_loop_min, input_s0_y_loop_extent, input_s0_x_loop_min, input_s0_x_loop_extent]->{input_s0[input_s0_z, input_s0_y, input_s0_x]: "
                        "(input_s0_z_loop_min <= input_s0_z <= ((input_s0_z_loop_min + input_s0_z_loop_extent) + -1)) and (input_s0_y_loop_min <= input_s0_y <= ((input_s0_y_loop_min + input_s0_y_loop_extent) + -1)) and (input_s0_x_loop_min <= input_s0_x <= ((input_s0_x_loop_min + input_s0_x_loop_extent) + -1))}",
                        tiramisu::expr((uint8_t)val0), true, tiramisu::p_uint8, &test_reduction_operator);
    input_s0.set_access("{input_s0[input_s0_z, input_s0_y, input_s0_x]->buff_input[input_s0_z, input_s0_y, input_s0_x]}");

    // Define loop bounds for dimension "f_s0_y".
    tiramisu::constant f_s0_y_loop_min("f_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant f_s0_y_loop_extent("f_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);

    // Define loop bounds for dimension "f_s0_x".
    tiramisu::constant f_s0_x_loop_min("f_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant f_s0_x_loop_extent("f_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::computation f_s0("[f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_y, f_s0_x]: "
                        "(f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        tiramisu::expr((uint8_t)val1), true, tiramisu::p_uint8, &test_reduction_operator);
    f_s0.set_access("{f_s0[f_s0_y, f_s0_x]->buff_f[f_s0_y, f_s0_x]}");

    // Define loop bounds for dimension "f_s1_y".
    tiramisu::constant f_s1_y_loop_min("f_s1_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant f_s1_y_loop_extent("f_s1_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);

    // Define loop bounds for dimension "f_s1_x".
    tiramisu::constant f_s1_x_loop_min("f_s1_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant f_s1_x_loop_extent("f_s1_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);

    // Define loop bounds for dimension "f_s1_r4__x".
    tiramisu::constant f_s1_r4__x_loop_min("f_s1_r4__x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::constant f_s1_r4__x_loop_extent("f_s1_r4__x_loop_extent", tiramisu::expr((int32_t)10), tiramisu::p_int32, true, NULL, 0, &test_reduction_operator);
    tiramisu::computation f_s1("[f_s1_y_loop_min, f_s1_y_loop_extent, f_s1_x_loop_min, f_s1_x_loop_extent, f_s1_r4__x_loop_min, f_s1_r4__x_loop_extent]->{f_s1[f_s1_y, f_s1_x, f_s1_r4__x]: "
                        "(f_s1_y_loop_min <= f_s1_y <= ((f_s1_y_loop_min + f_s1_y_loop_extent) + -1)) and (f_s1_x_loop_min <= f_s1_x <= ((f_s1_x_loop_min + f_s1_x_loop_extent) + -1)) and (f_s1_r4__x_loop_min <= f_s1_r4__x <= ((f_s1_r4__x_loop_min + f_s1_r4__x_loop_extent) + -1))}",
                        tiramisu::expr(), true, tiramisu::p_uint8, &test_reduction_operator);
    f_s1.set_expression((f_s1(tiramisu::idx("f_s1_y"), tiramisu::idx("f_s1_x"), (tiramisu::idx("f_s1_r4__x") - tiramisu::expr((int32_t)1))) + input_s0(tiramisu::idx("f_s1_r4__x"), tiramisu::idx("f_s1_y"), tiramisu::idx("f_s1_x"))));
    f_s1.set_access("{f_s1[f_s1_y, f_s1_x, f_s1_r4__x]->buff_f[f_s1_y, f_s1_x]}");

    // Define compute level for "f".
    f_s0.after(input_s0, computation::root_dimension);
    f_s1.after(f_s0, computation::root_dimension);

    test_reduction_operator.set_arguments({&buff_f});
    test_reduction_operator.gen_time_processor_domain();
    test_reduction_operator.gen_isl_ast();
    test_reduction_operator.gen_halide_stmt();
    test_reduction_operator.dump_halide_stmt();
    test_reduction_operator.gen_halide_obj("build/generated_fct_test_09.o");
}


int main(int argc, char **argv)
{
    generate_function_1("test_reduction_operator", 20, 40, 13, 37);

    return 0;
}
