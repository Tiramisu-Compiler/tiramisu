#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("relu_layer_gpu");

    var i("i", 0, 4);
    input parameters("parameters", {i}, p_float32);

    constant C_N("C_N", cast(p_int32, parameters(0))); //input size 
    constant C_FIn("C_FIn", cast(p_int32, parameters(1))); //input features
    constant C_BATCH_SIZE("C_BATCH_SIZE", cast(p_int32, parameters(2)));

    var x("x", 0, C_N), y("y", 0, C_N), z("z", 0, C_FIn), n("n", 0, C_BATCH_SIZE);

    input c_input("c_input", {n, z, y, x}, p_float32);
    computation c_relu("c_relu", {n, z, y, x},
    expr(o_max, c_input(n, z, y, x),
    expr((float)0)) + parameters(3) * expr(o_min, c_input(n, z, y, x), expr((float)0)));

    // Layer II


    if (1)
    {
        c_relu.gpu_tile(y, x, 32, 32);

    }

    // Layer III

    buffer input_buff("input_buff", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_input);
    buffer parameters_buff("parameters_buff", {expr(4)}, p_float32, a_input);
    buffer output_buff("output_buff", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_output);

    buffer input_buff_gpu("input_buff_gpu", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_temporary, global::get_implicit_function(), "input_buff" );
    buffer parameters_buff_gpu("parameters_buff_gpu", {expr(4)}, p_float32, a_temporary, global::get_implicit_function(),"parameters_buff" );buffer output_buff_gpu("output_buff_gpu", {expr(C_BATCH_SIZE), expr(C_FIn), expr(C_N), expr(C_N)}, p_float32, a_temporary, global::get_implicit_function(), "output_buff");

    input_buff_gpu.tag_gpu_global();
    parameters_buff_gpu.tag_gpu_global();
    output_buff_gpu.tag_gpu_global();

    c_input.store_in(&input_buff_gpu);
    c_relu.store_in(&output_buff_gpu);
    parameters.store_in(&parameters_buff_gpu);

    tiramisu::codegen({&input_buff, &parameters_buff, &output_buff}, "relu_layer_generator_tiramisu.o", true);

    return 0;
}

    
