#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("relu_tiramisu");

    var i("i", 0, 4);
    input parameters("parameters", {i}, p_float32);

    constant C_N("C_N", cast(p_int32, parameters(0)));
    constant C_FIn("C_FIn", cast(p_int32, parameters(1)));
    constant C_BATCH_SIZE("C_BATCH_SIZE", cast(p_int32, parameters(2)));

    var x("x", 0, C_N), y("y", 0, C_N), z("z", 0, C_FIn), n("n", 0, C_BATCH_SIZE);

    input c_input("c_input", {n, z, y, x}, p_float32);
    computation c_relu("c_relu", {n, z, y, x}, expr(o_max, c_input(n, z, y, x), expr((float)0)) + parameters(3) * expr(o_min, c_input(n, z, y, x), expr((float)0)));

    // Layer II

    int o_block = 4;
    int vec_len = N / 4;
    int y_block = N / 4;

    if (LARGE_DATA_SET)
    {
        c_relu.tag_parallel_level(0);
        c_relu.split(3, vec_len);
        c_relu.tag_vector_level(4, vec_len);
        c_relu.tag_unroll_level(5);
    }
    else if (MEDIUM_DATA_SET)
    {
        c_relu.split(3, y_block);
        c_relu.tag_vector_level(5, vec_len);
        c_relu.tag_unroll_level(6);
    }
    else if (SMALL_DATA_SET)
    {
        c_relu.tag_parallel_level(0);
        c_relu.split(3, 32);
        c_relu.tag_vector_level(4, 32);
        c_relu.tag_unroll_level(5);
    }

    // Layer III

    buffer input_buff("input_buff", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_input);
    buffer parameters_buff("parameters_buff", {expr(4)}, p_float32, a_input);
    buffer output_buff("output_buff", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_output);

    c_input.store_in(&input_buff);
    c_relu.store_in(&output_buff);
    parameters.store_in(&parameters_buff);

    tiramisu::codegen({&input_buff, &parameters_buff, &output_buff}, "relu_layer_generator_tiramisu.o");

    return 0;
}
