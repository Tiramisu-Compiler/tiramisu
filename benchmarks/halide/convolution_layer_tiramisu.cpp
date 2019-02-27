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
    init("convolution_layer_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    function convolution_layer_tiramisu("convolution_layer_tiramisu");

    // parameters
    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]
    var i("i", 0, 5);
    input parameters("parameters",{i}, p_int32);

    constant C_N0("C_N0", parameters(0)+parameters(1));
    constant C_N("C_N", parameters(0));
    constant C_N1("C_N1", parameters(0)-parameters(1));    
    constant C_N2("C_N2", parameters(0)-parameters(1)-parameters(1));
    constant C_K("C_K", parameters(1)+expr(1));
    constant C_FIn("C_FIn", parameters(2));
    constant C_FOut("C_FOut", parameters(3));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4));


    var z1("z1", 0, C_FOut), k_x("k_x",0,C_K), k_y("k_y",0,C_K), k_z("k_z",0,C_FIn); // filter variables
    var x("x", 0, C_N0 ), y("y", 0, C_N0),  z("z", 0, C_FOut), n("n", 0, C_BATCH_SIZE ); // input
    var x1("x1", 0, C_N), y1("y1", 0, C_N); // conv


    // Input computations
    input c_input("c_input",{n, z, y, x} , p_float32);
    input bias("bias", {z1}, p_float32);
    input filter("filter", {z1, k_z , k_y, k_x}, p_float32);



    // First conv computations
    computation conv_init("conv_init",{n, z, y1, x1}, bias(z) );
    computation conv("conv",{n, z, y1, x1, k_z, k_y, k_x }, conv_init(n, z, y1, x1) + filter(z, k_z, k_y, k_x) * c_input(n, k_z, y1 + k_y, x1 + k_x));
     

   // Layer II

    int vec_len , y_block , o_block ;
  
    vec_len = 16;
    y_block = 8;
    o_block = 4;          
    


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

    // Layer III
    buffer parameters_buf("parameters_buf", {expr(5)}, p_int32, a_input);
    buffer input_buf("input_buf", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N0), expr(C_N0)}, p_float32, a_input);
    buffer conv_buf("conv_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N), expr(C_N)}, p_float32, a_output);
    buffer filter_buf("filter_buf", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_input);
    buffer bias_buf("bias_buf", {expr(C_FOut)}, p_float32, a_input);
    buffer conv2_buf("conv2_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N1), expr(C_N1)}, p_float32, a_output);
    buffer bias2_buf("bias2_buf", {expr(C_FOut)}, p_float32, a_input);
    buffer filter2_buf("filter2_buf", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_input);
    buffer maxpool_buf("maxpool_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N2), expr(C_N2)}, p_float32, a_output);

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf);

    bias.store_in(&bias_buf);
    filter.store_in(&filter_buf);
    conv_init.store_in(&conv_buf);
    conv.store_in(&conv_buf,{n, z, y1, x1});
    
    tiramisu::codegen({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf}, "build/generated_fct_convolution_layer.o");

    return 0;
}