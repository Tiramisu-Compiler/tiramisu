#include <tiramisu/tiramisu.h>

#include <string.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("conv_tiramisu");

    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]

    var i("i", 0, 5);
    input parameters("parameters",{i}, p_int32);

    constant N_INPUT("N_INPUT", parameters(0) + parameters(1)); // The input is padded, so its size is N + K instead of N.
    constant C_N("C_N", parameters(0));
    constant C_K("C_K", parameters(1));
    constant C_FIn("C_FIn", parameters(2));
    constant C_FOut("C_FOut", parameters(3));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4));

    var x("x", 0, C_N), y("y", 0, C_N), z("z", 0, C_FOut), n("n", 0, C_BATCH_SIZE ); // Iterators for the conv computations.
    var k_x("k_x", 0, C_K), k_y("k_y", 0, C_K), k_z("k_z", 0, C_FIn); // Iterators for the kernel (filter).

    // Input computations
    input c_input("c_input",{"n", "k_z", "y", "x"}, {C_BATCH_SIZE, C_FIn, N_INPUT, N_INPUT} , p_float32);
    input bias("bias", {"z"}, {C_FOut}, p_float32);
    input filter("filter", {z, k_z , k_y, k_x}, p_float32);

    // First conv computations
    computation conv_init("conv_init",{n, z, y, x}, bias(z));
    computation conv("conv",{n, z, y, x, k_z, k_y, k_x }, conv_init(n, z, y, x) + filter(z, k_z, k_y, k_x) * c_input(n, k_z, y + k_y, x + k_x));
    
    global::get_implicit_function()->add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5}"); // C_FIn%16=0 and C_N%16=0}");

    // Layer II
    int vec_len;
    int y_block;
    int o_block;

    if (LARGE_DATA_SET || MEDIUM_DATA_SET)
    {
            vec_len = 32;
            y_block = 32;
            o_block = 4;
    }
    if (SMALL_DATA_SET)
    {
            vec_len = 16;
            y_block = 8;
            o_block = 4;
    }

    conv_init.then(conv, y);

    conv_init.tag_parallel_level(n);

    // 0, 1,   2,   3,   4,   5,     6,
    // n, z,   y,   x, k_z, k_y,   k_x,
    conv.interchange(x, k_z);
    // n, z,   y, (k_z,   x), k_y, k_x,
    conv.interchange(y, k_z);
    // n, z, (k_z,  y),   x,  k_y, k_x,

    var z1("z1"), z2("z2");
    conv.split(z, o_block, z1, z2);
    conv_init.split(z, o_block, z1, z2);
    // n, (z1, z2), k_z,   y,   x, k_y, k_x,

    conv.split(3, y_block);
    conv.split(6, vec_len);
////    conv.tag_vector_level(7, vec_len);

    // n,  z1, z2,  k_z,  (y, y_t), x, k_y, k_x,
    conv_init.split(4, vec_len);
////    conv_init.tag_vector_level(5, vec_len);

    if (LARGE_DATA_SET)
    {
////	    conv.tag_unroll_level(8);
////	    conv.tag_unroll_level(9);
    }

    // Layer III
    buffer parameters_buf("parameters_buf", {expr(5)}, p_int32, a_input);
    buffer input_buf("input_buf", {expr(C_BATCH_SIZE), C_FIn, N_INPUT, N_INPUT}, p_float32, a_input);
    buffer conv_buf("conv_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N), expr(C_N)}, p_float32, a_output);
    buffer filter_buf("filter_buf", {expr(C_FOut), expr(C_FIn), expr(C_K), expr(C_K)}, p_float32, a_input);
    buffer bias_buf("bias_buf", {expr(C_FOut)}, p_float32, a_input);

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf);
    bias.store_in(&bias_buf);
    filter.store_in(&filter_buf);
    conv_init.store_in(&conv_buf);
    conv.store_in(&conv_buf,{n, z, y, x});

    tiramisu::codegen({&parameters_buf, &input_buf, &filter_buf, &bias_buf, &conv_buf},"generated_conv_layer.o");

    return 0;
}
