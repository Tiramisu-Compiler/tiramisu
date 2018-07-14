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

    tiramisu::function conv_fct("conv_tiramisu");

    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]
    tiramisu::computation parameters("{parameters[i]: 0<=i<=4}", tiramisu::expr(), false, p_int32, &conv_fct);
    tiramisu::constant C_L("C_L", NB_LAYERS, p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_T("C_T", NB_TIME_STEPS, p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_N("C_N", parameters(0), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_K("C_K", K, p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_FIn("C_FIn", parameters(2), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_FOut("C_FOut", parameters(3), p_int32, true, NULL, 0, &conv_fct);
    tiramisu::constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4), p_int32, true, NULL, 0, &conv_fct);

    tiramisu::var x("x"), y("y"), z("z"), n("n"), r_x("r_x"), r_y("r_y"), r_z("r_z");

    tiramisu::computation bias("[C_FOut]->{bias[z]: 0<=z<C_FOut}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);
    tiramisu::computation filter("[C_K, C_FIn, C_FOut]->{filter[z, r_z, r_y, r_x]: 0<=z<C_FOut and 0<=r_x<C_K and 0<=r_y<C_K and 0<=r_z<C_FIn}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);
    tiramisu::computation input("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{input[n, z, y, x]: 0<=x<C_N+C_K and 0<=y<C_N+C_K and 0<=z<C_FIn and 0<=n<C_BATCH_SIZE}", tiramisu::expr(), false, tiramisu::p_float32, &conv_fct);

    tiramisu::computation conv_init("[C_L, C_T, C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv_init[l, t, n, z, y, x]: 0<=l<C_L and 0<=t<C_T and 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE}", bias(z), true, tiramisu::p_float32, &conv_fct);

    tiramisu::expr c = conv_init(0, 0, n, z, y, x) + filter(z, r_z, r_y, r_x) * input(n, r_z, y + r_y, x + r_x);
    tiramisu::computation conv("[C_L, C_T, C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv[l, t, n, z, y, x, r_z, r_y, r_x]: 0<=l<C_L and 0<=t<C_T and 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE and 0<=r_x<C_K and 0<=r_y<C_K and 0<=r_z<C_FIn}", c, true, tiramisu::p_float32, &conv_fct);

    conv_fct.add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5 and C_FIn%16=0 and C_N%16=0}");

    // Layer II
    if (LARGE_DATA_SET)
    {
            int vec_len = 32;
            int y_block = 32;
            int o_block = 4;

	    conv_init.apply_transformation_on_schedule("{conv_init[0, 0, l, 0, t, 0, n, 0, z, 0, y, 0, x, 0]->conv_init[0, 0, l+t, 0, t, 0, n, 0, z, 0, y, 0, x, 0]}");
	    conv.apply_transformation_on_schedule("{conv[0, 0, l, 0, t, 0, n, 0, z, 0, y, 0, x, 0, r_z, 0, r_y, 0, r_x, 0]->conv[0, 0, l+t, 0, t, 0, n, 0, z, 0, y, 0, x, 0, r_z, 0, r_y, 0, r_x, 0]}");
//##            conv_init.tag_parallel_level(0+2);
            conv_init.tag_parallel_level(0+1);

            conv.after(conv_init, 2+2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3+2, 4+2);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3+2, 2+2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1+2, o_block);
            conv_init.split(1+2, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3+2, y_block);
            conv.split(6+2, vec_len);
            conv.tag_vector_level(7+2, vec_len);
	    conv.tag_unroll_level(8+2);
	    conv.tag_unroll_level(9+2);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4+2, vec_len);
            conv_init.tag_vector_level(5+2, vec_len);
    }
    else if (MEDIUM_DATA_SET)
    {
	    int vec_len = 32;
            int y_block = 32;
            int o_block = 4;

            conv_init.tag_parallel_level(0+2);
            conv.after(conv_init, 2+2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3+2, 4+2);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3+2, 2+2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1+2, o_block);
            conv_init.split(1+2, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3+2, y_block);
            conv.split(6+2, vec_len);
            conv.tag_vector_level(7+2, vec_len);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4+2, vec_len);
            conv_init.tag_vector_level(5+2, vec_len);
    }
    else if (SMALL_DATA_SET)
    {
            int vec_len = 16;
            int y_block = 8;
            int o_block = 4;

            conv_init.tag_parallel_level(0+2);
            conv.after(conv_init, 2+2);

            // 0, 1,   2,   3,   4,   5,     6,
            // n, z,   y,   x, r_z, r_y,   r_x,
            conv.interchange(3+2, 4+2);
            // n, z,   y, (r_z,   x), r_y,   r_x,
            conv.interchange(3+2, 2+2);
           // n, z, (r_z,   y),   x, r_y,   r_x,

            conv.split(1+2, o_block);
            conv_init.split(1+2, o_block);
            // n, (z, z_t), r_z,   y,       x, r_y,   r_x,

            conv.split(3+2, y_block);
            conv.split(6+2, vec_len);
            conv.tag_vector_level(7+2, vec_len);

            // n,  z, z_t,  r_z,  (y, y_t), x, r_y,   r_x,
            conv_init.split(4+2, vec_len);
            conv_init.tag_vector_level(5+2, vec_len);
    }

    // Layer III
    tiramisu::buffer parameters_buf("parameters_buf", {tiramisu::expr(5)}, tiramisu::p_int32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer input_buf("input_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FIn), tiramisu::expr(N+K), tiramisu::expr(N+K)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer conv_buf("conv_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FOut), tiramisu::expr(N), tiramisu::expr(N)}, tiramisu::p_float32, tiramisu::a_output, &conv_fct);
    tiramisu::buffer filter_buf("filter_buf", {tiramisu::expr(FOut), tiramisu::expr(FIn), tiramisu::expr(K), tiramisu::expr(K)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);
    tiramisu::buffer bias_buf("bias_buf", {tiramisu::expr(FIn)}, tiramisu::p_float32, tiramisu::a_input, &conv_fct);


    conv_init.set_access("{conv_init[l, t, n, z, y, x]->conv_buf[n, z, y, x]}");
    conv.set_access("{conv[l, t, n, z, y, x, r_x, r_y, r_z]->conv_buf[n, z, y, x]}");
    parameters.set_access("{parameters[i]->parameters_buf[i]}");
    input.set_access("{input[n, z, y, x]->input_buf[n, z, y, x]}");
    bias.set_access("{bias[z]->bias_buf[z]}");
    filter.set_access("{filter[z, r_z, r_y, r_x]->filter_buf[z, r_z, r_y, r_x]}");

    conv_fct.set_arguments({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf});
    conv_fct.gen_time_space_domain();
    conv_fct.gen_isl_ast();
    conv_fct.gen_halide_stmt();
    conv_fct.dump_halide_stmt();
    conv_fct.gen_halide_obj("generated_conv_tiramisu.o");

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
