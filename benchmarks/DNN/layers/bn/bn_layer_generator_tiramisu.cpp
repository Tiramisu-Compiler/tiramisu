/* 
    This benchmark calculates the batch normalization result on Input tensor 
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    Output[n,z,y,x] = ( Input[n,z,y,x] - mean[z] ) / sqrt ( variance[z] );
    
    mean and variance are calculated over the axes: n, y, and x
    
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    mean[z] += Input[n,z,y,x]/(BATCH_SIZE*N*N) ;
    
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    variance[z] += (Input[n,z,y,x]- mean[z])²/(BATCH_SIZE*N*N) ;    
*/

#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("bn_tiramisu");

    // N: parameters[0]
    // FIn: parameters[1]
    // BATCH_SIZE: parameters[2]

    var i("i", 0, 3);
    input parameters("parameters", {i}, p_int32);
    constant C_N("C_N", parameters(0));
    constant C_NX("C_NX", parameters(0));
    constant C_FIn("C_FIn", parameters(1));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(2));
    constant C_NB_ELEMENTS("C_NB_ELEMENTS", parameters(0) * parameters(0) * parameters(2));

    var x("x", 0, C_N), x1("x1", 1, C_N), y("y", 0, C_N), y1("y1", 1, C_N), z("z", 0, C_FIn), n("n", 0, C_BATCH_SIZE), n1("n1", 1, C_BATCH_SIZE);
    var x_m("x_m", C_N - 1, C_N), y_m("y_m", C_N - 1, C_N);

    input c_input("c_input", {n, z, y, x}, p_float32);

    computation init_mean("init_mean", {n, z, y, x}, c_input(n, z, y, x));
    computation x_mean("x_mean", {n, z, y, x1}, init_mean(n, z, y, x1) + init_mean(n, z, y, x1 - 1));
    computation y_mean("y_mean", {n, z, y1, x_m}, x_mean(n, z, y1, x_m) + x_mean(n, z, y1 - 1, x_m));
    computation mean("mean", {n1, z, y_m, x_m}, y_mean(n1, z, y_m, x_m) + y_mean(n1 - 1, z, y_m, x_m));

    computation init_variance("init_variance", {n, z, y, x}, (c_input(n, z, y, x) - expr(o_div, mean(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1), cast(p_float32, C_NB_ELEMENTS))) * (c_input(n, z, y, x) - expr(o_div, mean(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1), cast(p_float32, C_NB_ELEMENTS))));

    computation x_variance("x_variance", {n, z, y, x1}, init_variance(n, z, y, x1) + init_variance(n, z, y, x1 - 1));
    computation y_variance("y_variance", {n, z, y1, x_m}, x_variance(n, z, y1, x_m) + x_variance(n, z, y1 - 1, x_m));
    computation variance("variance", {n1, z, y_m, x_m}, y_variance(n1, z, y_m, x_m) + y_variance(n1 - 1, z, y_m, x_m));

    computation output("output", {n, z, y, x}, expr(o_div, (c_input(n, z, y, x) - expr(o_div, mean(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1), cast(p_float32, C_NB_ELEMENTS))), expr(o_sqrt, expr(o_div, variance(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1), cast(p_float32, C_NB_ELEMENTS)))));

    init_mean.tag_parallel_level(n);

    x_mean.tag_parallel_level(n);
    x_mean.after(init_mean, x1);

    y_mean.tag_parallel_level(n);
    y_mean.after(x_mean, y1);

    mean.after(y_mean, n1);

    init_variance.tag_parallel_level(n);
    init_variance.after(mean, computation::root_dimension);

    x_variance.tag_parallel_level(n);
    x_variance.after(init_variance, computation::root_dimension);

    y_variance.tag_parallel_level(n);
    y_variance.after(x_variance, computation::root_dimension);

    variance.after(y_variance, computation::root_dimension);

    output.tag_parallel_level(n);
    output.after(variance, computation::root_dimension);
    output.tag_unroll_level(x);

    buffer input_buf("input_buf", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_input);
    buffer parameters_buf("parameters_buf", {expr(3)}, p_int32, a_input);
    buffer output_buf("output_buf", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);
    buffer mean_buff("mean_buff", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);
    buffer variance_buff("variance_buff", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf);
    init_mean.store_in(&mean_buff);
    mean.store_in(&mean_buff);
    x_mean.store_in(&mean_buff);
    y_mean.store_in(&mean_buff);
    init_variance.store_in(&variance_buff);
    x_variance.store_in(&variance_buff);
    y_variance.store_in(&variance_buff);
    variance.store_in(&variance_buff);
    output.store_in(&output_buf);

    tiramisu::codegen({&input_buf, &parameters_buf, &mean_buff, &variance_buff, &output_buf}, "bn_layer_generator_tiramisu.o");
    return 0;
}
