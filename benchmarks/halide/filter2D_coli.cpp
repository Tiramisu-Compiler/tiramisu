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

    coli::function filter2D_coli("filter2D_coli");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int filter2D_extent_2 = SIZE2;
    int filter2D_extent_1 = SIZE1 - 8;
    int filter2D_extent_0 = SIZE0 - 8;
    coli::buffer buff_filter2D("buff_filter2D", 3, {coli::expr(filter2D_extent_2), coli::expr(filter2D_extent_1), coli::expr(filter2D_extent_0)}, coli::p_uint8, NULL, coli::a_output, &filter2D_coli);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 3, {coli::expr(input_extent_2), coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_uint8, NULL, coli::a_input, &filter2D_coli);
    coli::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_uint8, &filter2D_coli);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;
    coli::buffer buff_kernel("buff_kernel", 2, {coli::expr(kernel_extent_1), coli::expr(kernel_extent_0)}, coli::p_float32, NULL, coli::a_input, &filter2D_coli);
    coli::computation kernel("[kernel_extent_1, kernel_extent_0]->{kernel[i1, i0]: (0 <= i1 <= (kernel_extent_1 + -1)) and (0 <= i0 <= (kernel_extent_0 + -1))}", expr(), false, coli::p_float32, &filter2D_coli);
    kernel.set_access("{kernel[i1, i0]->buff_kernel[i1, i0]}");


    // Define loop bounds for dimension "filter2D_s0_c".
    coli::constant filter2D_s0_c_loop_min("filter2D_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::constant filter2D_s0_c_loop_extent("filter2D_s0_c_loop_extent", coli::expr(filter2D_extent_2), coli::p_int32, true, NULL, 0, &filter2D_coli);

    // Define loop bounds for dimension "filter2D_s0_y".
    coli::constant filter2D_s0_y_loop_min("filter2D_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::constant filter2D_s0_y_loop_extent("filter2D_s0_y_loop_extent", coli::expr(filter2D_extent_1), coli::p_int32, true, NULL, 0, &filter2D_coli);

    // Define loop bounds for dimension "filter2D_s0_x".
    coli::constant filter2D_s0_x_loop_min("filter2D_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::constant filter2D_s0_x_loop_extent("filter2D_s0_x_loop_extent", coli::expr(filter2D_extent_0), coli::p_int32, true, NULL, 0, &filter2D_coli);
    coli::computation filter2D_s0("[filter2D_s0_c_loop_min, filter2D_s0_c_loop_extent, filter2D_s0_y_loop_min, filter2D_s0_y_loop_extent, filter2D_s0_x_loop_min, filter2D_s0_x_loop_extent]->{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]: "
                        "(filter2D_s0_c_loop_min <= filter2D_s0_c <= ((filter2D_s0_c_loop_min + filter2D_s0_c_loop_extent) + -1)) and (filter2D_s0_y_loop_min <= filter2D_s0_y <= ((filter2D_s0_y_loop_min + filter2D_s0_y_loop_extent) + -1)) and (filter2D_s0_x_loop_min <= filter2D_s0_x <= ((filter2D_s0_x_loop_min + filter2D_s0_x_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_uint8, (((((((((coli::expr((float)0) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)0)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)0))))*kernel(coli::expr((int32_t)0), coli::expr((int32_t)0)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)0)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)1))))*kernel(coli::expr((int32_t)0), coli::expr((int32_t)1)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)0)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)2))))*kernel(coli::expr((int32_t)0), coli::expr((int32_t)2)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)1)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)0))))*kernel(coli::expr((int32_t)1), coli::expr((int32_t)0)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)1)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)1))))*kernel(coli::expr((int32_t)1), coli::expr((int32_t)1)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)1)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)2))))*kernel(coli::expr((int32_t)1), coli::expr((int32_t)2)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)2)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)0))))*kernel(coli::expr((int32_t)2), coli::expr((int32_t)0)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)2)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)1))))*kernel(coli::expr((int32_t)2), coli::expr((int32_t)1)))) + (coli::expr(coli::o_cast, coli::p_float32, input(coli::idx("filter2D_s0_c"), (coli::idx("filter2D_s0_y") + coli::expr((int32_t)2)), (coli::idx("filter2D_s0_x") + coli::expr((int32_t)2))))*kernel(coli::expr((int32_t)2), coli::expr((int32_t)2))))), true, coli::p_uint8, &filter2D_coli);
    filter2D_s0.set_access("{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->buff_filter2D[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]}");

    // Define compute level for "filter2D".
    filter2D_s0.first(computation::root_dimension);

    // Add schedules.
    filter2D_s0.tag_parallel_dimension(1);

    filter2D_coli.set_arguments({&buff_input, &buff_kernel, &buff_filter2D});
    filter2D_coli.gen_time_processor_domain();
    filter2D_coli.gen_isl_ast();
    filter2D_coli.gen_halide_stmt();
    filter2D_coli.dump_halide_stmt();
    filter2D_coli.gen_halide_obj("build/generated_fct_filter2D.o");

    return 0;
}

