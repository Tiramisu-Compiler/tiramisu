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

    init("conv_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    function conv_fct("conv_tiramisu");
   

    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]

    var i("i", 0, 5);
    input parameters("parameters",{i}, p_int32);

    constant C_N("C_N", parameters(0)+ parameters(1));
    constant C_N1("C_N1",parameters(0));
    constant C_K("C_K", parameters(1));
    constant C_FIn("C_FIn", parameters(2));
    constant C_FOut("C_FOut", parameters(3));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4));

    var x("x", 0, C_N ), y("y", 0, C_N),  z("z", 0, C_FOut), n("n", 0, C_BATCH_SIZE ); // input
    var k_x("k_x",0,C_K), k_y("k_y",0,C_K), k_z("k_z",0,C_FIn); // filter variables
    var x1("x1", 0, C_N1), y1("y1", 0, C_N1); // conv

    // Input computations
    input c_input("c_input",{n, k_z, y, x} , p_float32);
    input bias("bias", {z}, p_float32);
    input filter("filter", {z, k_z , k_y, k_x}, p_float32);

    // First conv computations
    computation conv_init("conv_init",{n, z, y1, x1}, bias(z) );
    computation conv("conv",{n, z, y1, x1, k_z, k_y, k_x }, conv_init(n, z, y1, x1) + filter(z, k_z, k_y, k_x) * c_input(n, k_z, y1 + k_y, x1 + k_x));
    
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
    buffer parameters_buf("parameters_buf", {expr(5)}, p_int32, a_input);
    buffer input_buf("input_buf", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_input);
    buffer conv_buf("conv_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N1), expr(C_N1)}, p_float32, a_output);
    buffer filter_buf("filter_buf", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_input);
    buffer bias_buf("bias_buf", {expr(C_FOut)}, p_float32, a_input);

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf);
    bias.store_in(&bias_buf);
    filter.store_in(&filter_buf);
    conv_init.store_in(&conv_buf);
    conv.store_in(&conv_buf,{n, z, y1, x1});

    tiramisu::codegen({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf},"generated_conv_tiramisu.o");

    return 0;
}

#if 0
int main(int argc, char **argv)
{
    ImageParam            input{Float(32), 4, "input"};
    ImageParam            filter{Float(32), 4, "filter"};
    ImageParam            bias{Float(32), 1, "bias"};

    Var x("x"), y("y"), z("z"), n("n");

    Func f_conv("conv");
    RDom r(0, K, 0, K, 0, FIn);

    f_conv(x, y, z, n) = bias(z);
    f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);

    /* THE SCHEDULE */
    if (SCHEDULE_CPU)
    {
	if (LARGE_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 128;
	    int o_block_size = 8;
	    int y_block = 32;
	    f_conv.compute_root();
	    f_conv.fuse(z, n, par).parallel(par);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);
	}
	else if (MEDIUM_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 32;
	    int o_block_size = 4;
	    int y_block = 32;
	    f_conv.compute_root();
	    f_conv.fuse(z, n, par).parallel(par);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);
	}
	else if (SMALL_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 32;
	    int o_block_size = 8;
	    int y_block = 32;
	    f_conv.compute_root();
	    f_conv.parallel(n);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);
	}
    }
 
    Halide::Target target = Halide::get_host_target();

    f_conv.compile_to_object("generated_conv.o",
                             {input, filter, bias},
                             "conv_halide",
                             target);
    return 0;
}
#endif
