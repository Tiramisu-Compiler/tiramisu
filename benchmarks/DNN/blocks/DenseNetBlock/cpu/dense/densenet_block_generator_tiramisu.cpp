#include <tiramisu/tiramisu.h>
#include <vector>
#include "configure.h"

using namespace tiramisu;

int main()
{
    init("densenet_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), z("z", 0, 4*GR), n("n", 0, BATCH_SIZE);

    var x_pad("x", 0, N + 2), y_pad("y", 0, N + 2);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y);

    var fin_b("fin_b", 0, FIN_NB_BLOCKS), ffin("ffin", 0, FIN_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, fin_b, y_pad, x_pad, ffin}, p_float32);
    input bn_scale("bn_scale", {fin_b, ffin}, p_float32);
    input bn_shift("bn_shift", {fin_b, ffin}, p_float32);

    input conv_filter("conv_filter", {fin_b, fout_b, k_y, k_x, ffin, ffout}, p_float32);
    input conv_bias("conv_bias", {fout_b, ffout}, p_float32);

    // Batch normalization followed by ReLU
    // Compute the sum over the features dimension (z)
    computation input_sum_init("input_sum_init", {fin_b, ffin}, cast(p_float32, 0));
    computation input_sum(
        "input_sum", 
        {fin_b, n, y_pad, x_pad, ffin}, 
        input_sum_init(fin_b, ffin) + c_input(n, fin_b, y_pad, x_pad, ffin)
    );

    // Compute the sum of squares over the features dimension (z)
    computation input_sum_squares_init("input_sum_squares_init", {fin_b, ffin}, cast(p_float32, 0));
    computation input_sum_squares(
        "input_sum_squares", 
        {fin_b, n, y_pad, x_pad, ffin}, 
        input_sum_squares_init(fin_b, ffin) + c_input(n, fin_b, y_pad, x_pad, ffin) * c_input(n, fin_b, y_pad, x_pad, ffin)
    );

    computation input_mean(
        "input_mean", 
        {fin_b, ffin}, 
        input_sum(fin_b, 0, 0, 0, ffin) / cast(p_float32, BATCH_SIZE*(N+2)*(N+2))
    );

    computation input_sd(
        "input_sd", 
        {fin_b, ffin}, 
        expr(
            o_sqrt, 
            input_sum_squares(fin_b, 0, 0, 0, ffin) / cast(p_float32, BATCH_SIZE*(N+2)*(N+2)) - input_mean(fin_b, ffin) * input_mean(fin_b, ffin) + cast(p_float32, EPSILON)
        )
    );
    
    // Compute BN followed by ReLU
    computation bn(
        "bn", 
        {n, fin_b, y_pad, x_pad, ffin}, 
        bn_scale(fin_b, ffin) * ((c_input(n, fin_b, y_pad, x_pad, ffin) - input_mean(fin_b, ffin)) / input_sd(fin_b, ffin)) + bn_shift(fin_b, ffin)
    );

    computation relu(
        "relu", 
        {n, fin_b, y_pad, x_pad, ffin}, 
        expr(
            o_max, 
            cast(p_float32, 0), 
            bn(n, fin_b, y_pad, x_pad, ffin)
        )
    );

    // Convolution computation
    computation init_output("init_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fin_b, fout_b, y, x, k_y, k_x, ffin, ffout}, 
        init_output(n, fout_b, y, x, ffout) + relu(n, fin_b, y + k_y, x + k_x, ffin)*conv_filter(fin_b, fout_b, k_y, k_x, ffin, ffout)
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    input_sum_init.then(input_sum_squares_init, ffin)
                  .then(input_sum, fin_b)
                  .then(input_sum_squares, ffin)
                  .then(input_mean, fin_b)
                  .then(input_sd, ffin)
                  .then(init_output, computation::root)
                  .then(bn, n)
                  .then(relu, ffin)
                  .then(conv, fin_b);

    input_sum.vectorize(ffin, FIN_BLOCKING);
    input_sum.tag_parallel_level(fin_b);

    var y_b, x_b;
    var yy, xx;

    conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        
    // n, fin_b, fout_b, y_b, x_b, yy, xx, k_y, k_x, ffin, ffout
    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    // n, fin_b, fout_b, y_b, x_b, yy, k_y, k_x, xx, ffin, ffout
    conv.interchange(yy, k_y);
    conv.interchange(yy, k_x);
    // n, fin_b, fout_b, y_b, x_b, k_y, k_x, yy, xx, ffin, ffout
        
    conv.tag_parallel_level(n);
    conv.vectorize(ffout, VEC_LEN);
    bn.vectorize(ffin, FIN_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    buffer input_mean_buf("input_mean_buf", {FIN_NB_BLOCKS, FIN_BLOCKING}, p_float32, a_input);
    buffer input_sd_buf("input_sd_buf", {FIN_NB_BLOCKS, FIN_BLOCKING}, p_float32, a_input);

    buffer workspace_buf("workspace_buf", {BATCH_SIZE, N + 2, N + 2, FIN_BLOCKING}, p_float32, a_input);

    input_sum_init.store_in(&input_mean_buf);
    input_sum.store_in(&input_mean_buf, {fin_b, ffin});
    input_mean.store_in(&input_mean_buf);
    
    input_sum_squares_init.store_in(&input_sd_buf);
    input_sum_squares.store_in(&input_sd_buf, {fin_b, ffin});
    input_sd.store_in(&input_sd_buf);

    bn.store_in(&workspace_buf, {n, y_pad, x_pad, ffin});
    relu.store_in(&workspace_buf, {n, y_pad, x_pad, ffin});

    init_output.store_in(&output_buf);
    conv.store_in(&output_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        bn_scale.get_buffer(), 
        bn_shift.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(),
        &input_mean_buf,
        &input_sd_buf,
        &workspace_buf,
        &output_buf
    }, "densenet_block_tiramisu.o");

    return 0;
}
