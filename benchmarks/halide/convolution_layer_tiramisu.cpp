#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/utils.h>

#include <string.h>
#include "configure.h"

#define SCHEDULE_CPU 1

using namespace tiramisu;

int main(int argc, char **argv)
{
    // set default tiramisu options.
    global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function conv_fct("convolution_layer_tiramisu");

    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]

    tiramisu::computation parameters("{parameters[i]: 0<=i<=4}", tiramisu::expr(), false, p_int32, &conv_fct);
    tiramisu::constant C_N("C_N", parameters(0), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_K("C_K", K, p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_FIn("C_FIn", parameters(2), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_FOut("C_FOut", parameters(3), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4), p_int32, true, NULL, 0, &conv_fct);

    tiramisu::var x("x"), y("y"), z("z"), n("n"), r_x("r_x"), r_y("r_y"), r_z("r_z");
    tiramisu::computation bias("[C_FOut]->{bias[z]: 0<=z<C_FOut}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);
    tiramisu::computation filter("[C_K, C_FIn, C_FOut]->{filter[z, r_z, r_y, r_x]: 0<=z<C_FOut and 0<=r_x<C_K and 0<=r_y<C_K and 0<=r_z<C_FIn}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);
    tiramisu::computation input("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{input[n, z, y, x]: 0<=x<C_N+C_K and 0<=y<C_N+C_K and 0<=z<C_FIn and 0<=n<C_BATCH_SIZE}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);

    tiramisu::computation conv_init("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv_init[n, z, y, x]: 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE}", bias(z), true, tiramisu::p_float32, &conv_fct);

    tiramisu::expr c = conv_init(n, z, y, x) + filter(z, r_z, r_y, r_x) * input(n, r_z, y + r_y, x + r_x);
    tiramisu::computation conv("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv[n, z, y, x, r_z, r_y, r_x]: 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE and 0<=r_x<C_K and 0<=r_y<C_K and 0<=r_z<C_FIn}", c, true, tiramisu::p_float32, &conv_fct);

    conv_fct.add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5 and C_FIn%16=0 and C_N%16=0}");

    // Layer II
    if (LARGE_DATA_SET)
    {
	if (0) {
		int vec_len = 32;
		int y_block = 32;
		int o_block = 4;

		conv_init.tag_parallel_level(0);
		conv_init.split(3, vec_len);
		conv_init.tag_vector_level(4, vec_len);

		conv.after(conv_init, 1);

		conv.interchange(3, 4);
		// conv_init.tile(1, 2, 32, 32);
		// conv.tile(1, 2, 32, 32);

		// conv.split(1, o_block);
		conv.split(2, y_block);
		conv.split(5, vec_len);
		conv.tag_vector_level(6, vec_len);
		conv.tag_unroll_level(7);
		conv.tag_unroll_level(8);
	}
	if (0) {
	    int vec_len = 32;
	    int y_block = 32;
	    int o_block = 4;

	    conv_init.tag_parallel_level(0);
	    conv.after(conv_init, 2);

	    // 0, 1,   2,   3,   4,   5,     6,
	    // n, z,   y,   x, r_z, r_y,   r_x,
	    conv.interchange(3, 4);
	    // n, z,   y  r_z,   x, r_y,   r_x,

	    conv.split(1, o_block);
	    conv_init.split(1, o_block);

	    conv.split(3, y_block);
	    conv.split(6, vec_len);
	    conv.tag_vector_level(7, vec_len);
	    conv.tag_unroll_level(9);

	    conv_init.split(4, vec_len);
	    conv_init.tag_vector_level(5, vec_len);
	}
	if (1)
	{
            int vec_len = 32;
            int y_block = 32;
            int o_block = 4;

            conv_init.tag_parallel_level(0);
            conv.after(conv_init, 2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3, 4);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3, 2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1, o_block);
            conv_init.split(1, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3, y_block);
            conv.split(6, vec_len);
            conv.tag_vector_level(7, vec_len);
	    conv.tag_unroll_level(8);
	    conv.tag_unroll_level(9);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4, vec_len);
            conv_init.tag_vector_level(5, vec_len);
	}
    }
    else if (MEDIUM_DATA_SET)
    {
	if (0) // Replicate of Halide's schedule. But Halide = 18ms and Tiramisu=54ms, so there is a problem in Tiramisu code generator.
	{
	    int vec_len = 32;
	    int y_block = 32;
	    int o_block = 4;

	    conv_init.tag_parallel_level(0);
	    conv.after(conv_init, tiramisu::computation::root_dimension);

	    // 0, 1,   2,   3,   4,   5,     6,
	    // n, z,   y,   x, r_z, r_y,   r_x,
	    conv.interchange(3, 4);
	    // n, z,   y, (r_z,   x), r_y,   r_x,
	    conv.interchange(3, 2);
	    // n, z, (r_z,   y),   x, r_y,   r_x,

	    conv.split(1, o_block);
	    // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

	    conv.split(4, y_block);
	    // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,

	    conv.interchange(2, 3);
	    // n, z, (r_z, z_t), y, y_t, x, r_y,   r_x,

	    conv.interchange(3, 4);
	    // n, z, r_z, (y, z_t), y_t, x, r_y,   r_x,

	    conv.split(6, vec_len); 
	    conv.tag_vector_level(7, vec_len);
	    conv.tag_unroll_level(8);
	    conv.tag_unroll_level(9);

	    conv_init.split(3, vec_len);
	    conv_init.tag_vector_level(4, vec_len);
	}
	if (1)
	{
            int vec_len = 32;
            int y_block = 32;
            int o_block = 4;

            conv_init.tag_parallel_level(0);
            conv.after(conv_init, 2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3, 4);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3, 2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1, o_block);
            conv_init.split(1, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3, y_block);
            conv.split(6, vec_len);
            conv.tag_vector_level(7, vec_len);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4, vec_len);
            conv_init.tag_vector_level(5, vec_len);
	}
    }
    else if (SMALL_DATA_SET)
    {
            int vec_len = 16;
            int y_block = 8;
            int o_block = 4;

            conv_init.tag_parallel_level(0);
            conv.after(conv_init, 2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3, 4);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3, 2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1, o_block);
            conv_init.split(1, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3, y_block);
            conv.split(6, vec_len);
            conv.tag_vector_level(7, vec_len);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4, vec_len);
            conv_init.tag_vector_level(5, vec_len);
    }

    // Layer III
    tiramisu::buffer parameters_buf("parameters_buf", {tiramisu::expr(5)}, tiramisu::p_int32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer input_buf("input_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FIn), tiramisu::expr(N+K), tiramisu::expr(N+K)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer conv_buf("conv_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FOut), tiramisu::expr(N), tiramisu::expr(N)}, tiramisu::p_float32, tiramisu::a_output, &conv_fct);
    tiramisu::buffer filter_buf("filter_buf", {tiramisu::expr(FOut), tiramisu::expr(FIn), tiramisu::expr(K), tiramisu::expr(K)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer bias_buf("bias_buf", {tiramisu::expr(FIn)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);


    conv_init.set_access("{conv_init[n, z, y, x]->conv_buf[n, z, y, x]}");
    conv.set_access("{conv[n, z, y, x, r_x, r_y, r_z]->conv_buf[n, z, y, x]}");
    parameters.set_access("{parameters[i]->parameters_buf[i]}");
    input.set_access("{input[n, z, y, x]->input_buf[n, z, y, x]}");
    bias.set_access("{bias[z]->bias_buf[z]}");
    filter.set_access("{filter[z, r_z, r_y, r_x]->filter_buf[z, r_z, r_y, r_x]}");

    conv_fct.set_arguments({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf});
    conv_fct.gen_time_space_domain();
    conv_fct.gen_isl_ast();
    conv_fct.gen_halide_stmt();
    conv_fct.dump_halide_stmt();
    conv_fct.gen_halide_obj("build/generated_fct_convolution_layer.o");

    return 0;
}