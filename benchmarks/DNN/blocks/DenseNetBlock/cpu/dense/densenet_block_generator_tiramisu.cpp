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

    computation conv_init("conv_init", {n, y, fout_b, x, ffout}, conv_bias(fout_b, ffout));
    view conv_out("conv_out", {n, y, x, fout_b, ffout}, p_float32);

    // Convolution computation
    // x_bound is used to have the width dimension divisible by X_BLOCKING
    // in the conv computation.
    var x_bound("x_bound", 0, X_BOUND);
    var x_conclude("x_conclude", X_BOUND, N);

    // Compute convolution from 0 to x_bound
    computation conv(
        "conv",
        {n, fin_b, y, x_bound, k_y, k_x, ffin, fout_b, ffout},
        conv_out(n, y, x_bound, fout_b, ffout) + conv_filter(fin_b, fout_b, k_y, k_x, ffin, ffout) * relu(n, fin_b, y + k_y, x_bound + k_x, ffin)
    );

    // Compute convolution from x_bound to N
    computation conv_conclude(
        "conv_conclude",
        {n, fin_b, y, k_y, k_x, ffin, fout_b, ffout, x_conclude},
        conv_out(n, y, x_conclude, fout_b, ffout) + conv_filter(fin_b, fout_b, k_y, k_x, ffin, ffout) * relu(n, fin_b, y + k_y, x_conclude + k_x, ffin)
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    input_sum.vectorize(ffin, VEC_LEN);
    input_sum.tag_parallel_level(fin_b);

    bn.vectorize(ffin, VEC_LEN);

    /*
     * Schedule for conv
     */
    // We introduce those two computations to do register blocking
    computation reg_load(
        "reg_load",
        {n, fin_b, y, x_bound, fout_b, ffout},
        conv_init(n, y, fout_b, x_bound, ffout)
    );

    computation reg_store(
        "reg_store",
        {n, fin_b, y, x_bound, fout_b, ffout},
        conv(n, fin_b, y, x_bound, 0, 0, 0, fout_b, ffout)
    );

    // Split over dimension x
    var x_b, xx;
    conv.split(x_bound, X_BLOCKING, x_b, xx);

    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    conv.interchange(xx, ffin);
    conv.interchange(xx, fout_b);
    conv.interchange(xx, ffout);

    reg_load.split(x_bound, X_BLOCKING, x_b, xx);
    reg_store.split(x_bound, X_BLOCKING, x_b, xx);

    reg_load.interchange(xx, fout_b);
    reg_load.interchange(xx, ffout);

    reg_store.interchange(xx, fout_b);
    reg_store.interchange(xx, ffout);

    // Vectorize and unroll
    reg_load.vectorize(ffout, VEC_LEN);
    conv.vectorize(ffout, VEC_LEN);
    reg_store.vectorize(ffout, VEC_LEN);

    conv.tag_unroll_level(xx);
    conv.tag_unroll_level(fout_b);

    reg_load.tag_unroll_level(xx);
    reg_load.tag_unroll_level(fout_b);

    reg_store.tag_unroll_level(xx);
    reg_store.tag_unroll_level(fout_b);

    // schedule for conv_conclude
    // This schedule is the same as conv computation
    computation reg_load_conclude(
        "reg_load_conclude",
        {n, fin_b, y, fout_b, ffout, x_conclude},
        conv_init(n, y, fout_b, x_conclude, ffout)
    );

    computation reg_store_conclude(
        "reg_store_conclude",
        {n, fin_b, y, fout_b, ffout, x_conclude},
        conv_conclude(n, fin_b, y, 0, 0, 0, fout_b, ffout, x_conclude)
    );

    reg_load_conclude.vectorize(ffout, VEC_LEN);
    conv_conclude.vectorize(ffout, VEC_LEN);
    reg_store_conclude.vectorize(ffout, VEC_LEN);

    conv_conclude.tag_unroll_level(x_conclude);
    conv_conclude.tag_unroll_level(fout_b);

    reg_load_conclude.tag_unroll_level(x_conclude);
    reg_load_conclude.tag_unroll_level(fout_b);

    reg_store_conclude.tag_unroll_level(x_conclude);
    reg_store_conclude.tag_unroll_level(fout_b);

    // Parallelize and order
    conv.tag_parallel_level(n);

    input_sum_init.then(input_sum_squares_init, ffin)
                  .then(input_sum, fin_b)
                  .then(input_sum_squares, ffin)
                  .then(input_mean, fin_b)
                  .then(input_sd, ffin)
                  .then(conv_init, computation::root)
                  .then(bn, n)
                  .then(relu, ffin)
                  .then(reg_load, fin_b)
                  .then(conv, x_b)
                  .then(reg_store, x_b)
                  .then(reg_load_conclude, y)
                  .then(conv_conclude, y)
                  .then(reg_store_conclude, y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    buffer input_mean_buf("input_mean_buf", {FIN_NB_BLOCKS, FIN_BLOCKING}, p_float32, a_input);
    buffer input_sd_buf("input_sd_buf", {FIN_NB_BLOCKS, FIN_BLOCKING}, p_float32, a_input);

    buffer workspace_buf("workspace_buf", {BATCH_SIZE, N + 2, N + 2, FIN_BLOCKING}, p_float32, a_input);

    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {FOUT_NB_BLOCKS, X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    input_sum_init.store_in(&input_mean_buf);
    input_sum.store_in(&input_mean_buf, {fin_b, ffin});
    input_mean.store_in(&input_mean_buf);
    
    input_sum_squares_init.store_in(&input_sd_buf);
    input_sum_squares.store_in(&input_sd_buf, {fin_b, ffin});
    input_sd.store_in(&input_sd_buf);

    bn.store_in(&workspace_buf, {n, y_pad, x_pad, ffin});
    relu.store_in(&workspace_buf, {n, y_pad, x_pad, ffin});

    /*
     * Storage for conv
     */
    conv_init.store_in(&output_buf, {n, fout_b, y, x, ffout});
    conv_out.store_in(&reg_buf, {fout_b, x%X_BLOCKING, ffout});

    reg_load.store_in(&reg_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    conv.store_in(&reg_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    reg_store.store_in(&output_buf, {n, fout_b, y, x_bound, ffout});

    reg_load_conclude.store_in(&reg_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv_conclude.store_in(&reg_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    reg_store_conclude.store_in(&output_buf, {n, fout_b, y, x_conclude, ffout});

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
