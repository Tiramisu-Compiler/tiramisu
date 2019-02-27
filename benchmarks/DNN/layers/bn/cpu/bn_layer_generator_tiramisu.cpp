/* 
    This benchmark calculates the batch normalization result on Input tensor 
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    Output[n,z,y,x] = ( Input[n,z,y,x] - sum[z] ) / sqrt ( sumSquared[z] );
    
    sum and sumSquared are calculated over the axes: n, y, and x
    
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    sum[z] += Input[n,z,y,x]/(BATCH_SIZE*N*N) ;
    
    for z = 0 .. FIn
        for n = 0 .. BATCH_SIZE
            for y = 0 .. N
                for x = 0 .. N
                    sumSquared[z] += (Input[n,z,y,x]- sum[z])²/(BATCH_SIZE*N*N) ;    
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

    //Calculate the sum of the input values in parallel
    computation init_sum("init_sum", {n, z, y, x}, c_input(n, z, y, x));
    computation x_sum("x_sum", {n, z, y, x1}, init_sum(n, z, y, x1) + init_sum(n, z, y, x1 - 1));
    computation y_sum("y_sum", {n, z, y1, x_m}, x_sum(n, z, y1, x_m) + x_sum(n, z, y1 - 1, x_m));
    computation sum("sum", {n1, z, y_m, x_m}, y_sum(n1, z, y_m, x_m) + y_sum(n1 - 1, z, y_m, x_m));

    //Calculate the sum of the input values squared in parallel
    computation init_sumSquared("init_sumSquared", {n, z, y, x}, c_input(n, z, y, x) * c_input(n, z, y, x));
    computation x_sumSquared("x_sumSquared", {n, z, y, x1}, init_sumSquared(n, z, y, x1) + init_sumSquared(n, z, y, x1 - 1));
    computation y_sumSquared("y_sumSquared", {n, z, y1, x_m}, x_sumSquared(n, z, y1, x_m) + x_sumSquared(n, z, y1 - 1, x_m));
    computation sumSquared("sumSquared", {n1, z, y_m, x_m}, y_sumSquared(n1, z, y_m, x_m) + y_sumSquared(n1 - 1, z, y_m, x_m));

    computation output("output", {n, z, y, x}, (c_input(n, z, y, x) - sum(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1) / cast(p_float32, C_NB_ELEMENTS)) / expr(o_sqrt, sumSquared(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1) / cast(p_float32, C_NB_ELEMENTS) - (sum(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1) / cast(p_float32, C_NB_ELEMENTS)) * (sum(C_BATCH_SIZE - 1, z, C_N - 1, C_NX - 1) / cast(p_float32, C_NB_ELEMENTS))));

    init_sum.tag_parallel_level(n);

    x_sum.tag_parallel_level(n);
    x_sum.after(init_sumSquared, x1);

    y_sum.tag_parallel_level(n);
    y_sum.after(x_sumSquared, y1);

    sum.after(y_sumSquared, n1);

    init_sumSquared.tag_parallel_level(n);
    init_sumSquared.after(init_sum, x);

    x_sumSquared.tag_parallel_level(n);
    x_sumSquared.after(x_sum, x1);

    y_sumSquared.tag_parallel_level(n);
    y_sumSquared.after(y_sum, y1);

    sumSquared.after(sum, n1);

    output.after(sumSquared, n);
    output.tag_unroll_level(n);

    buffer input_buf("input_buf", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_input);
    buffer parameters_buf("parameters_buf", {expr(3)}, p_int32, a_input);
    buffer output_buf("output_buf", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);
    buffer sum_buff("sum_buff", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);
    buffer sumSquared_buff("sumSquared_buff", {C_BATCH_SIZE, C_FIn, C_N, C_N}, p_float32, a_output);

    parameters.store_in(&parameters_buf);
    c_input.store_in(&input_buf);
    init_sum.store_in(&sum_buff);
    sum.store_in(&sum_buff);
    x_sum.store_in(&sum_buff);
    y_sum.store_in(&sum_buff);
    init_sumSquared.store_in(&sumSquared_buff);
    x_sumSquared.store_in(&sumSquared_buff);
    y_sumSquared.store_in(&sumSquared_buff);
    sumSquared.store_in(&sumSquared_buff);
    output.store_in(&output_buf);

    tiramisu::codegen({&input_buf, &parameters_buf, &sum_buff, &sumSquared_buff, &output_buf}, "bn_layer_generator_tiramisu.o");
    return 0;
}
