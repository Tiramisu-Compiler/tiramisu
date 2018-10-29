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
    tiramisu::function conv_block("conv_block_tiramisu");

    // convolution parameter
    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]
    tiramisu::computation parameters("{parameters[i]: 0<=i<=8}", tiramisu::expr(), false, p_int32, &conv_block);
    tiramisu::constant C_N("C_N", parameters(0), p_int32, true, NULL, 0, &conv_block);
    tiramisu::constant C_K("C_K", parameters(1), p_int32, true, NULL, 0, &conv_block);
    tiramisu::constant C_FIn("C_FIn", parameters(2), p_int32, true, NULL, 0, &conv_block);
    tiramisu::constant C_FOut("C_FOut", parameters(3), p_int32, true, NULL, 0, &conv_block);
    tiramisu::constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4), p_int32, true, NULL, 0, &conv_block);
    

    tiramisu::var x("x"), y("y"), z("z"), n("n"), r_x("r_x"), r_y("r_y"), r_z("r_z");

    // Input computations
    tiramisu::computation negative_slope("{negative_slope[i]: 0<=i<1}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);
    tiramisu::computation bias("[C_FOut]->{bias[z]: 0<=z<C_FOut}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);
    tiramisu::computation filter("[C_K, C_FIn, C_FOut]->{filter[z, r_z, r_y, r_x]: 0<=z<C_FOut and 0<=r_x<=C_K and 0<=r_y<=C_K and 0<=r_z<C_FIn}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);
    tiramisu::computation input("[C_N, C_K, C_FIn, C_BATCH_SIZE]->{input[n, z, y, x]: 0<=x<C_N+C_K and 0<=y<C_N+C_K and 0<=z<C_FIn and 0<=n<C_BATCH_SIZE}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);

    tiramisu::computation bias2("[C_FOut]->{bias2[z]: 0<=z<C_FOut}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);
    tiramisu::computation filter2("[C_K, C_FIn, C_FOut]->{filter2[z, r_z, r_y, r_x]: 0<=z<C_FOut and 0<=r_x<=C_K and 0<=r_y<=C_K and 0<=r_z<C_FIn}", tiramisu::expr(), false, tiramisu::p_float32, &conv_block);


    // First conv computations
    tiramisu::computation conv_init("[C_N, C_FOut, C_BATCH_SIZE]->{conv_init[n, z, y, x]: 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE}", bias(z), true, tiramisu::p_float32, &conv_block);
    tiramisu::expr c = conv_init(n, z, y, x) + filter(z, r_z, r_y, r_x) * input(n, r_z, y + r_y, x + r_x);
    tiramisu::computation conv("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv[n, z, y, x, r_z, r_y, r_x]: 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE and 0<=r_x<=C_K and 0<=r_y<=C_K and 0<=r_z<C_FIn}", c , true, tiramisu::p_float32, &conv_block);
     
    //first relu
    tiramisu::computation relu("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{relu[n, z, y, x]: 0<=x<C_N and 0<=y<C_N and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE }", expr(o_max,conv(n, z, y, x, 0, 0, 0), expr((float)  0))+ negative_slope(0)*expr(o_min,conv(n, z, y, x, 0, 0, 0), expr((float)  0)), true, tiramisu::p_float32, &conv_block);


    // Second conv computations
    tiramisu::computation conv2_init("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv2_init[n, z, y, x]: 0<=x<C_N-C_K and 0<=y<C_N-C_K and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE}", bias2(z), true, tiramisu::p_float32, &conv_block);
    tiramisu::expr c2 = conv2_init(n, z, y, x) + filter2(z, r_z, r_y, r_x) * relu(n, r_z, y + r_y, x + r_x);
    tiramisu::computation conv2("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{conv2[n, z, y, x, r_z, r_y, r_x]: 0<=x<C_N-C_K and 0<=y<C_N-C_K and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE and 0<=r_x<=C_K and 0<=r_y<=C_K and 0<=r_z<C_FIn}", c2, true, tiramisu::p_float32, &conv_block);


    //second relu
    tiramisu::computation relu2("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{relu2[n, z, y, x]: 0<=x<C_N-C_K and 0<=y<C_N-C_K and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE }", expr(o_max,conv2(n, z, y, x, 0, 0, 0), expr((float)  0))+ negative_slope(0)*expr(o_min,conv2(n, z, y, x, 0, 0, 0), expr((float)  0)), true, tiramisu::p_float32, &conv_block);


    //maxpooling computation
    tiramisu::computation maxpool_init("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{maxpool_init[n, z, y, x]: 0<=x<C_N-C_K-C_K and 0<=y<C_N-C_K-C_K and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE}", expr((float) -2147483647), true, tiramisu::p_float32, &conv_block);
     tiramisu::computation maxpool("[C_N, C_K, C_FOut, C_FIn, C_BATCH_SIZE]->{maxpool[n, z, y, x, r_y, r_x]: 0<=x<C_N-C_K-C_K and 0<=y<C_N-C_K-C_K and 0<=z<C_FOut and 0<=n<C_BATCH_SIZE and 0<=r_x<=C_K and 0<=r_y<=C_K }", expr(), true, p_float32, &conv_block);
    
    maxpool.set_expression( expr(o_max, maxpool_init(n, z, y, x),relu2(n, z, y + r_y, x + r_x))); 

    //conv_block.add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5 and C_FIn%16=0 and C_N%16=0}");

	


    // Layer II

            int vec_len , y_block , o_block ;
    if (LARGE_DATA_SET)
    {
            vec_len = 32;
            y_block = 32;
            o_block = 4;

    }
    else if (MEDIUM_DATA_SET)
    {
            vec_len = 32;
            y_block = 32;
            o_block = 4;

    }
    else if (SMALL_DATA_SET)
    {
            vec_len = 16;
            y_block = 8;
            o_block = 4;          
    }

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


	    // Order between the first and second convolutions
	    relu.after(conv, tiramisu::computation::root_dimension);
	    conv2_init.after(relu, tiramisu::computation::root_dimension);


	    // Schedule of 2nd convolution
	    conv2_init.tag_parallel_level(0);
            conv2.after(conv2_init, 2);
	    relu2.after(conv2,2);


	    // Schedule of maxpooling
	    maxpool_init.after(relu2, tiramisu::computation::root_dimension);
	    maxpool.after(maxpool_init, tiramisu::computation::root_dimension);


    // Layer III
    tiramisu::buffer parameters_buf("parameters_buf", {tiramisu::expr(5)}, tiramisu::p_int32, tiramisu::a_input, &conv_block);
    tiramisu::buffer input_buf("input_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FIn), tiramisu::expr(N+K), tiramisu::expr(N+K)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);
    tiramisu::buffer conv_buf("conv_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FOut), tiramisu::expr(N), tiramisu::expr(N)}, tiramisu::p_float32, tiramisu::a_output, &conv_block);
    tiramisu::buffer filter_buf("filter_buf", {tiramisu::expr(FOut), tiramisu::expr(FIn), tiramisu::expr(K), tiramisu::expr(K)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);
    tiramisu::buffer bias_buf("bias_buf", {tiramisu::expr(FOut)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);
 
   tiramisu::buffer conv2_buf("conv2_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FOut), tiramisu::expr(N-K), tiramisu::expr(N-K)}, tiramisu::p_float32, tiramisu::a_output, &conv_block);
    tiramisu::buffer bias2_buf("bias2_buf", {tiramisu::expr(FOut)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);
   tiramisu::buffer filter2_buf("filter2_buf", {tiramisu::expr(FOut), tiramisu::expr(FIn), tiramisu::expr(K), tiramisu::expr(K)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);

    tiramisu::buffer negative_slope_buf("negative_slope_buf", {tiramisu::expr(1)}, tiramisu::p_float32, tiramisu::a_input, &conv_block);

    tiramisu::buffer maxpool_buf("maxpool_buf", {tiramisu::expr(BATCH_SIZE), tiramisu::expr(FOut), tiramisu::expr(N-2*K), tiramisu::expr(N-2*K)}, tiramisu::p_float32, tiramisu::a_output, &conv_block);


    conv_init.set_access("{conv_init[n, z, y, x]->conv_buf[n, z, y, x]}");
    conv.set_access("{conv[n, z, y, x, r_x, r_y, r_z]->conv_buf[n, z, y, x]}");
    relu.set_access("{relu[n, z, y, x]->conv_buf[n, z, y, x]}");
    conv2_init.set_access("{conv2_init[n, z, y, x]->conv2_buf[n, z, y, x]}");
    conv2.set_access("{conv2[n, z, y, x, r_x, r_y, r_z]->conv2_buf[n, z, y, x]}");
    bias2.set_access("{bias2[z]->bias2_buf[z]}");
    filter2.set_access("{filter2[z, r_z, r_y, r_x]->filter2_buf[z, r_z, r_y, r_x]}");

    relu2.set_access("{relu2[n, z, y, x]->conv2_buf[n, z, y, x]}");
    maxpool_init.set_access("{maxpool_init[n, z, y, x]->maxpool_buf[n, z, y, x]}");
    maxpool.set_access("{maxpool[n, z, y, x, r_x, r_y]->maxpool_buf[n, z, y, x]}");
    negative_slope.set_access("{negative_slope[i]->negative_slope_buf[i]}");

    parameters.set_access("{parameters[i]->parameters_buf[i]}");
    input.set_access("{input[n, z, y, x]->input_buf[n, z, y, x]}");
    bias.set_access("{bias[z]->bias_buf[z]}");
    filter.set_access("{filter[z, r_z, r_y, r_x]->filter_buf[z, r_z, r_y, r_x]}");
    negative_slope.set_access("{negative_slope[0]->negative_slope_buf[0]}");
    

    conv_block.set_arguments({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf, &filter2_buf, &bias2_buf, &conv2_buf,&maxpool_buf,&negative_slope_buf});
    conv_block.gen_time_space_domain();
    conv_block.gen_isl_ast();
    conv_block.gen_halide_stmt();
    conv_block.dump_halide_stmt();
    conv_block.gen_halide_obj("generated_conv_block.o");

    return 0;
}
