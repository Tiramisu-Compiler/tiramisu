/* 
    This benchmark represents a ResNet Block which includes the next layers in order:
    - convolution layer
    - 2D batch normalization
    - relu
    - convolution 
    - 2D batch normalization
    Each convolution function is fused with the BN layer that follows it.
*/

#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("fused_resnet_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), n("n", 0, BATCH_SIZE);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y);

    var fin_b("fin_b", 0, FIN_NB_BLOCKS), ffin("ffin", 0, FIN_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);
    
    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);

    // x_bound is used to have the width dimension divisible by X_BLOCKING
    // in the conv computations.
    var x_bound("x_bound", 0, X_BOUND);
    var x_conclude("x_conclude", X_BOUND, N);

    input c_input("c_input", {n, fin_b, y_pad, x_pad, ffin}, p_float32);

    input filter1("filter1", {fout_b, fin_b, k_y, k_x, ffin, ffout}, p_float32);
    input bias1("bias1", {fout_b, ffout}, p_float32);
    input bn1_scale("bn1_scale", {fout_b, ffout}, p_float32);
    input bn1_shift("bn1_shift", {fout_b, ffout}, p_float32);

    input filter2("filter2", {fout_b, fin_b, k_y, k_x, ffin, ffout}, p_float32);
    input bias2("bias2", {fout_b, ffout}, p_float32);
    input bn2_scale("bn2_scale", {fout_b, ffout}, p_float32);
    input bn2_shift("bn2_shift", {fout_b, ffout}, p_float32);

    /*
     * First convolution computations
     */
    computation zero_conv1("zero_conv1", {n, fout_b, y_pad, x_pad, ffout}, cast(p_float32, 0));

    // Compute convolution from 0 to x_bound
    computation conv1_init("conv1_init", {n, y, x_bound, fout_b, ffout}, bias1(fout_b, ffout));
    computation conv1(
        "conv1",
        {n, y, x_bound, fin_b, k_y, k_x, ffin, fout_b, ffout},
        conv1_init(n, y, x_bound, fout_b, ffout) + filter1(fout_b, fin_b, k_y, k_x, ffin, ffout) * c_input(n, fin_b, y + k_y, x_bound + k_x, ffin)
    );

    // Compute convolution from x_bound to N
    computation conv1_init_conclude("conv1_init_conclude", {n, y, fout_b, ffout, x_conclude}, bias1(fout_b, ffout));
    computation conv1_conclude(
        "conv1_conclude",
        {n, y, fin_b, k_y, k_x, ffin, fout_b, ffout, x_conclude},
        conv1_init_conclude(n, y, fout_b, ffout, x_conclude) + filter1(fout_b, fin_b, k_y, k_x, ffin, ffout) * c_input(n, fin_b, y + k_y, x_conclude + k_x, ffin)
    );

    /*
     * BN-ReLU computations
     */
    view conv1_result("conv1_result", {n, fout_b, y, x, ffout}, p_float32);

    // Compute the sum over the features dimension (z)
    computation input1_sum_init("input1_sum_init", {fout_b, ffout}, cast(p_float32, 0));
    computation input1_sum(
        "input1_sum", 
        {fout_b, n, y, x, ffout}, 
        input1_sum_init(fout_b, ffout) + conv1_result(n, fout_b, y, x, ffout)
    );

    // Compute the sum of squares over the features dimension (z)
    computation input1_sum_squares_init("input1_sum_squares_init", {fout_b, ffout}, cast(p_float32, 0));
    computation input1_sum_squares(
        "input1_sum_squares", 
        {fout_b, n, y, x, ffout}, 
        input1_sum_squares_init(fout_b, ffout) + conv1_result(n, fout_b, y, x, ffout) * conv1_result(n, fout_b, y, x, ffout)
    );

    computation input1_mean(
        "input1_mean", 
        {fout_b, ffout}, 
        input1_sum(fout_b, 0, 0, 0, ffout) / cast(p_float32, BATCH_SIZE*N*N)
    );

    computation input1_sd(
        "input1_sd", 
        {fout_b, ffout}, 
        expr(
            o_sqrt, 
            input1_sum_squares(fout_b, 0, 0, 0, ffout) / cast(p_float32, BATCH_SIZE*N*N) - input1_mean(fout_b, ffout) * input1_mean(fout_b, ffout) + cast(p_float32, EPSILON)
        )
    );

    // Compute BN followed by ReLU
    computation bn1(
        "bn1", 
        {n, fout_b, y, x, ffout}, 
        bn1_scale(fout_b, ffout) * ((conv1_result(n, fout_b, y, x, ffout) - input1_mean(fout_b, ffout)) / input1_sd(fout_b, ffout)) + bn1_shift(fout_b, ffout)
    );

    computation relu1(
        "relu1", 
        {n, fout_b, y, x, ffout}, 
        expr(
            o_max, 
            cast(p_float32, 0), 
            bn1(n, fout_b, y, x, ffout)
        )
    );

    /*
     * Second convolution computations
     */
    view relu1_padded("relu1_padded", {n, fin_b, y_pad, x_pad, ffin}, p_float32);

    // Compute convolution from 0 to x_bound
    computation conv2_init("conv2_init", {n, y, x_bound, fout_b, ffout}, bias2(fout_b, ffout));
    computation conv2(
        "conv2",
        {n, y, x_bound, fin_b, k_y, k_x, ffin, fout_b, ffout},
        conv2_init(n, y, x_bound, fout_b, ffout) + filter2(fout_b, fin_b, k_y, k_x, ffin, ffout) * relu1_padded(n, fin_b, y + k_y, x_bound + k_x, ffin)
    );

    // Compute convolution from x_bound to N
    computation conv2_init_conclude("conv2_init_conclude", {n, y, fout_b, ffout, x_conclude}, bias2(fout_b, ffout));
    computation conv2_conclude(
        "conv2_conclude",
        {n, y, fin_b, k_y, k_x, ffin, fout_b, ffout, x_conclude},
        conv2_init_conclude(n, y, fout_b, ffout, x_conclude) + filter2(fout_b, fin_b, k_y, k_x, ffin, ffout) * relu1_padded(n, fin_b, y + k_y, x_conclude + k_x, ffin)
    );

    /*
     * BN computations
     */
    view conv2_result("conv2_result", {n, fout_b, y, x, ffout}, p_float32);

    // Compute the sum over the features dimension (z)
    computation input2_sum_init("input2_sum_init", {fout_b, ffout}, cast(p_float32, 0));
    computation input2_sum(
        "input2_sum", 
        {fout_b, n, y, x, ffout}, 
        input2_sum_init(fout_b, ffout) + conv2_result(n, fout_b, y, x, ffout)
    );

    // Compute the sum of squares over the features dimension (z)
    computation input2_sum_squares_init("input2_sum_squares_init", {fout_b, ffout}, cast(p_float32, 0));
    computation input2_sum_squares(
        "input2_sum_squares", 
        {fout_b, n, y, x, ffout}, 
        input2_sum_squares_init(fout_b, ffout) + conv2_result(n, fout_b, y, x, ffout) * conv2_result(n, fout_b, y, x, ffout)
    );

    computation input2_mean(
        "input2_mean", 
        {fout_b, ffout}, 
        input2_sum(fout_b, 0, 0, 0, ffout) / cast(p_float32, BATCH_SIZE*N*N)
    );

    computation input2_sd(
        "input2_sd", 
        {fout_b, ffout}, 
        expr(
            o_sqrt, 
            input2_sum_squares(fout_b, 0, 0, 0, ffout) / cast(p_float32, BATCH_SIZE*N*N) - input2_mean(fout_b, ffout) * input2_mean(fout_b, ffout) + cast(p_float32, EPSILON)
        )
    );

    computation bn2(
        "bn2", 
        {n, fout_b, y, x, ffout}, 
        bn2_scale(fout_b, ffout) * ((conv2_result(n, fout_b, y, x, ffout) - input2_mean(fout_b, ffout)) / input2_sd(fout_b, ffout)) + bn2_shift(fout_b, ffout)
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var x_b, xx;

    /* 
     * schedule for the first conv computation
     */

    // We introduce those two computations to do register blocking
    computation reg1_store(
        "reg1_store",
        {n, y, x_bound, fout_b, ffout},
        conv1(n, y, x_bound, 0, 0, 0, 0, fout_b, ffout)
    );

    // Split over dimension x
    conv1.split(x_bound, X_BLOCKING, x_b, xx);

    conv1.interchange(xx, fin_b);
    conv1.interchange(xx, k_y);
    conv1.interchange(xx, k_x);
    conv1.interchange(xx, ffin);
    conv1.interchange(xx, fout_b);
    conv1.interchange(xx, ffout);

    conv1_init.split(x_bound, X_BLOCKING, x_b, xx);
    reg1_store.split(x_bound, X_BLOCKING, x_b, xx);

    conv1_init.interchange(xx, fout_b);
    conv1_init.interchange(xx, ffout);

    reg1_store.interchange(xx, fout_b);
    reg1_store.interchange(xx, ffout);

    // Vectorize and unroll
    conv1_init.vectorize(ffout, FOUT_BLOCKING);
    conv1.vectorize(ffout, FOUT_BLOCKING);
    reg1_store.vectorize(ffout, FOUT_BLOCKING);

    conv1.tag_unroll_level(xx);
    conv1.tag_unroll_level(fout_b);

    conv1_init.tag_unroll_level(xx);
    conv1_init.tag_unroll_level(fout_b);

    reg1_store.tag_unroll_level(xx);
    reg1_store.tag_unroll_level(fout_b);

    // schedule for conv1_conclude
    // This schedule is the same as conv1 computation
    computation reg1_store_conclude(
        "reg1_store_conclude",
        {n, y, fout_b, ffout, x_conclude},
        conv1_conclude(n, y, 0, 0, 0, 0, fout_b, ffout, x_conclude)
    );

    conv1_init_conclude.vectorize(ffout, FOUT_BLOCKING);
    conv1_conclude.vectorize(ffout, FOUT_BLOCKING);
    reg1_store_conclude.vectorize(ffout, FOUT_BLOCKING);

    conv1_conclude.tag_unroll_level(x_conclude);
    conv1_conclude.tag_unroll_level(fout_b);

    conv1_init_conclude.tag_unroll_level(x_conclude);
    conv1_init_conclude.tag_unroll_level(fout_b);

    reg1_store_conclude.tag_unroll_level(x_conclude);
    reg1_store_conclude.tag_unroll_level(fout_b);

    /* 
     * schedule for BN-ReLU computations
     */
    input1_sum.vectorize(ffout, FOUT_BLOCKING);
    input1_sum.tag_parallel_level(fout_b);

    bn1.tag_parallel_level(n);
    bn1.vectorize(ffout, FOUT_BLOCKING);

    /* 
     * schedule for the second conv computation
     */
    // We introduce those two computations to do register blocking
    computation reg2_store(
        "reg2_store",
        {n, y, x_bound, fout_b, ffout},
        conv2(n, y, x_bound, 0, 0, 0, 0, fout_b, ffout)
    );

    // Split over dimension x
    conv2.split(x_bound, X_BLOCKING, x_b, xx);

    conv2.interchange(xx, fin_b);
    conv2.interchange(xx, k_y);
    conv2.interchange(xx, k_x);
    conv2.interchange(xx, ffin);
    conv2.interchange(xx, fout_b);
    conv2.interchange(xx, ffout);

    conv2_init.split(x_bound, X_BLOCKING, x_b, xx);
    reg2_store.split(x_bound, X_BLOCKING, x_b, xx);

    conv2_init.interchange(xx, fout_b);
    conv2_init.interchange(xx, ffout);

    reg2_store.interchange(xx, fout_b);
    reg2_store.interchange(xx, ffout);

    // Vectorize and unroll
    conv2_init.vectorize(ffout, FOUT_BLOCKING);
    conv2.vectorize(ffout, FOUT_BLOCKING);
    reg2_store.vectorize(ffout, FOUT_BLOCKING);

    conv2.tag_unroll_level(xx);
    conv2.tag_unroll_level(fout_b);

    conv2_init.tag_unroll_level(xx);
    conv2_init.tag_unroll_level(fout_b);

    reg2_store.tag_unroll_level(xx);
    reg2_store.tag_unroll_level(fout_b);

    // schedule for conv_conclude
    // This schedule is the same as conv computation
    computation reg2_store_conclude(
        "reg2_store_conclude",
        {n, y, fout_b, ffout, x_conclude},
        conv2_conclude(n, y, 0, 0, 0, 0, fout_b, ffout, x_conclude)
    );

    conv2_init_conclude.vectorize(ffout, FOUT_BLOCKING);
    conv2_conclude.vectorize(ffout, FOUT_BLOCKING);
    reg2_store_conclude.vectorize(ffout, FOUT_BLOCKING);

    conv2_conclude.tag_unroll_level(x_conclude);
    conv2_conclude.tag_unroll_level(fout_b);

    conv2_init_conclude.tag_unroll_level(x_conclude);
    conv2_init_conclude.tag_unroll_level(fout_b);

    reg2_store_conclude.tag_unroll_level(x_conclude);
    reg2_store_conclude.tag_unroll_level(fout_b);

    /* 
     * schedule for the second BN computation
     */
    input2_sum.vectorize(ffout, FOUT_BLOCKING);
    input2_sum.tag_parallel_level(fout_b);

    bn2.tag_parallel_level(n);
    bn2.vectorize(ffout, FOUT_BLOCKING);

    /*
     * Parallelize and order
     */
    conv1.tag_parallel_level(n);
    conv2.tag_parallel_level(n);
 
    zero_conv1.then(conv1_init, n)
              .then(conv1, x_b)
              .then(reg1_store, x_b)
              .then(conv1_init_conclude, y)
              .then(conv1_conclude, y)
              .then(reg1_store_conclude, y)
              .then(input1_sum_init, computation::root)
              .then(input1_sum_squares_init, ffout)
              .then(input1_sum, fout_b)
              .then(input1_sum_squares, ffout)
              .then(input1_mean, fout_b)
              .then(input1_sd, ffout)
              .then(bn1, computation::root)
              .then(relu1, ffout)
              .then(conv2_init, n)
              .then(conv2, x_b)
              .then(reg2_store, x_b)
              .then(conv2_init_conclude, y)
              .then(conv2_conclude, y)
              .then(reg2_store_conclude, y)
              .then(input2_sum_init, computation::root)
              .then(input2_sum_squares_init, ffout)
              .then(input2_sum, fout_b)
              .then(input2_sum_squares, ffout)
              .then(input2_mean, fout_b)
              .then(input2_sd, ffout)
              .then(bn2, computation::root);;

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv1_buf("conv1_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N + 2, N + 2, FOUT_BLOCKING}, p_float32, a_input);
    buffer conv2_buf("conv2_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    buffer input_mean_buf("input_mean_buf", {FOUT_NB_BLOCKS, FOUT_BLOCKING}, p_float32, a_input);
    buffer input_sd_buf("input_sd_buf", {FOUT_NB_BLOCKS, FOUT_BLOCKING}, p_float32, a_input);

    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg1_buf("reg1_buf", {FOUT_NB_BLOCKS, X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);
    buffer reg2_buf("reg2_buf", {FOUT_NB_BLOCKS, X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);
    
    /*
     * First convolution storage
     */
    zero_conv1.store_in(&conv1_buf);

    conv1_init.store_in(&reg1_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    conv1.store_in(&reg1_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    reg1_store.store_in(&conv1_buf, {n, fout_b, y + 1, x_bound + 1, ffout});

    conv1_init_conclude.store_in(&reg1_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv1_conclude.store_in(&reg1_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    reg1_store_conclude.store_in(&conv1_buf, {n, fout_b, y + 1, x_conclude + 1, ffout});

    /*
     * BN-ReLU storage
     */
    conv1_result.store_in(&conv1_buf, {n, fout_b, y + 1, x + 1, ffout});

    input1_sum_init.store_in(&input_mean_buf);
    input1_sum.store_in(&input_mean_buf, {fout_b, ffout});
    input1_mean.store_in(&input_mean_buf);
    
    input1_sum_squares_init.store_in(&input_sd_buf);
    input1_sum_squares.store_in(&input_sd_buf, {fout_b, ffout});
    input1_sd.store_in(&input_sd_buf);

    bn1.store_in(&conv1_buf, {n, fout_b, y + 1, x + 1, ffout});
    relu1.store_in(&conv1_buf, {n, fout_b, y + 1, x + 1, ffout});

    /*
     * Second convolution storage
     */
    relu1_padded.store_in(&conv1_buf);

    conv2_init.store_in(&reg2_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    conv2.store_in(&reg2_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    reg2_store.store_in(&conv2_buf, {n, fout_b, y, x_bound, ffout});

    conv2_init_conclude.store_in(&reg2_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv2_conclude.store_in(&reg2_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    reg2_store_conclude.store_in(&conv2_buf, {n, fout_b, y, x_conclude, ffout});

    /*
     * BN storage
     */
    conv2_result.store_in(&conv2_buf, {n, fout_b, y, x, ffout});

    input2_sum_init.store_in(&input_mean_buf);
    input2_sum.store_in(&input_mean_buf, {fout_b, ffout});
    input2_mean.store_in(&input_mean_buf);
    
    input2_sum_squares_init.store_in(&input_sd_buf);
    input2_sum_squares.store_in(&input_sd_buf, {fout_b, ffout});
    input2_sd.store_in(&input_sd_buf);

    bn2.store_in(&conv2_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(), 
        filter1.get_buffer(), 
        bias1.get_buffer(), 
        bn1_scale.get_buffer(),
        bn1_shift.get_buffer(),
        filter2.get_buffer(),
        bias2.get_buffer(),
        bn2_scale.get_buffer(),
        bn2_shift.get_buffer(),
        &input_mean_buf,
        &input_sd_buf,
        &conv1_buf,
        &conv2_buf
    }, "fused_resnet_block_generator_tiramisu.o");

    return 0;
}