#define __TIRAMISU_GENERATOR__
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
    input bn1_mean("bn1_mean", {fout_b, ffout}, p_float32);
    input bn1_variance("bn1_variance", {fout_b, ffout}, p_float32);

    input filter2("filter2", {fout_b, fin_b, k_y, k_x, ffin, ffout}, p_float32);
    input bias2("bias2", {fout_b, ffout}, p_float32);

    input bn2_scale("bn2_scale", {fout_b, ffout}, p_float32);
    input bn2_shift("bn2_shift", {fout_b, ffout}, p_float32);
    input bn2_mean("bn2_mean", {fout_b, ffout}, p_float32);
    input bn2_variance("bn2_variance", {fout_b, ffout}, p_float32);

    /*
     * Preliminary computations
     */
    computation bn1_alpha(
        "bn1_alpha",
        {fout_b, ffout},
        bn1_scale(fout_b, ffout) / expr(o_sqrt, bn1_variance(fout_b, ffout) + cast(p_float32, EPSILON))
    );

    computation bn1_beta(
        "bn1_beta",
        {fout_b, ffout},
        bn1_shift(fout_b, ffout) - (bn1_scale(fout_b, ffout) * bn1_mean(fout_b, ffout)) / expr(o_sqrt, bn1_variance(fout_b, ffout) + cast(p_float32, EPSILON))
    );

    computation bn2_alpha(
        "bn2_alpha",
        {fout_b, ffout},
        bn2_scale(fout_b, ffout) / expr(o_sqrt, bn2_variance(fout_b, ffout) + cast(p_float32, EPSILON))
    );

    computation bn2_beta(
        "bn2_beta",
        {fout_b, ffout},
        bn2_shift(fout_b, ffout) - (bn2_scale(fout_b, ffout) * bn2_mean(fout_b, ffout)) / expr(o_sqrt, bn2_variance(fout_b, ffout) + cast(p_float32, EPSILON))
    );

    computation zero_conv1("zero_conv1", {n, fout_b, y_pad, x_pad, ffout}, cast(p_float32, 0));

    /*
     * First convolution computations
     */

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
    computation bn1(
        "bn1",
        {n, y, x_bound, fout_b, ffout},
        bn1_alpha(fout_b, ffout) * conv1(n, y, x_bound, 0, 0, 0, 0, fout_b, ffout) + bn1_beta(fout_b, ffout)
    );

    computation relu1(
        "relu1",
        {n, y, x_bound, fout_b, ffout},
        expr(
            o_max,
            cast(p_float32, 0),
            bn1(n, y, x_bound, fout_b, ffout)
        )
    );

    computation bn1_conclude(
        "bn1_conclude",
        {n, y, fout_b, ffout, x_conclude},
        bn1_alpha(fout_b, ffout) * conv1_conclude(n, y, 0, 0, 0, 0, fout_b, ffout, x_conclude) + bn1_beta(fout_b, ffout)
    );

    computation relu1_conclude(
        "relu1_conclude",
        {n, y, fout_b, ffout, x_conclude},
        expr(
            o_max,
            cast(p_float32, 0),
            bn1_conclude(n, y, fout_b, ffout, x_conclude)
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
    computation bn2(
        "bn2",
        {n, y, x_bound, fout_b, ffout},
        bn2_alpha(fout_b, ffout) * conv2(n, y, x_bound, 0, 0, 0, 0, fout_b, ffout) + bn2_beta(fout_b, ffout)
    );

    computation bn2_conclude(
        "bn2_conclude",
        {n, y, fout_b, ffout, x_conclude},
        bn2_alpha(fout_b, ffout) * conv2_conclude(n, y, 0, 0, 0, 0, fout_b, ffout, x_conclude) + bn2_beta(fout_b, ffout)
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var x_b, xx;

    /*
     * schedule for the first conv computation
     */

    // Split over dimension x
    conv1.split(x_bound, X_BLOCKING, x_b, xx);

    conv1.interchange(xx, fin_b);
    conv1.interchange(xx, k_y);
    conv1.interchange(xx, k_x);
    conv1.interchange(xx, ffin);
    conv1.interchange(xx, fout_b);
    conv1.interchange(xx, ffout);

    conv1_init.split(x_bound, X_BLOCKING, x_b, xx);
    bn1.split(x_bound, X_BLOCKING, x_b, xx);
    relu1.split(x_bound, X_BLOCKING, x_b, xx);

    conv1_init.interchange(xx, fout_b);
    conv1_init.interchange(xx, ffout);

    bn1.interchange(xx, fout_b);
    bn1.interchange(xx, ffout);

    relu1.interchange(xx, fout_b);
    relu1.interchange(xx, ffout);

    // Vectorize and unroll
    conv1_init.vectorize(ffout, FOUT_BLOCKING);
    conv1.vectorize(ffout, FOUT_BLOCKING);
    bn1.vectorize(ffout, FOUT_BLOCKING);

    conv1.tag_unroll_level(xx);
    conv1.tag_unroll_level(fout_b);

    conv1_init.tag_unroll_level(xx);
    conv1_init.tag_unroll_level(fout_b);

    bn1.tag_unroll_level(xx);
    bn1.tag_unroll_level(fout_b);

    // schedule for conv1_conclude
    // This schedule is the same as conv1 computation
    conv1_init_conclude.vectorize(ffout, FOUT_BLOCKING);
    conv1_conclude.vectorize(ffout, FOUT_BLOCKING);
    bn1_conclude.vectorize(ffout, FOUT_BLOCKING);

    conv1_conclude.tag_unroll_level(x_conclude);
    conv1_conclude.tag_unroll_level(fout_b);

    conv1_init_conclude.tag_unroll_level(x_conclude);
    conv1_init_conclude.tag_unroll_level(fout_b);

    bn1_conclude.tag_unroll_level(x_conclude);
    bn1_conclude.tag_unroll_level(fout_b);

    /*
     * schedule for the second conv computation
     */

    // Split over dimension x
    conv2.split(x_bound, X_BLOCKING, x_b, xx);

    conv2.interchange(xx, fin_b);
    conv2.interchange(xx, k_y);
    conv2.interchange(xx, k_x);
    conv2.interchange(xx, ffin);
    conv2.interchange(xx, fout_b);
    conv2.interchange(xx, ffout);

    conv2_init.split(x_bound, X_BLOCKING, x_b, xx);
    bn2.split(x_bound, X_BLOCKING, x_b, xx);

    conv2_init.interchange(xx, fout_b);
    conv2_init.interchange(xx, ffout);

    bn2.interchange(xx, fout_b);
    bn2.interchange(xx, ffout);

    // Vectorize and unroll
    conv2_init.vectorize(ffout, FOUT_BLOCKING);
    conv2.vectorize(ffout, FOUT_BLOCKING);
    bn2.vectorize(ffout, FOUT_BLOCKING);

    conv2.tag_unroll_level(xx);
    conv2.tag_unroll_level(fout_b);

    conv2_init.tag_unroll_level(xx);
    conv2_init.tag_unroll_level(fout_b);

    bn2.tag_unroll_level(xx);
    bn2.tag_unroll_level(fout_b);

    // schedule for conv_conclude
    // This schedule is the same as conv computation
    conv2_init_conclude.vectorize(ffout, FOUT_BLOCKING);
    conv2_conclude.vectorize(ffout, FOUT_BLOCKING);
    bn2_conclude.vectorize(ffout, FOUT_BLOCKING);

    conv2_conclude.tag_unroll_level(x_conclude);
    conv2_conclude.tag_unroll_level(fout_b);

    conv2_init_conclude.tag_unroll_level(x_conclude);
    conv2_init_conclude.tag_unroll_level(fout_b);

    bn2_conclude.tag_unroll_level(x_conclude);
    bn2_conclude.tag_unroll_level(fout_b);

    /*
     * Parallelize and order
     */
    conv2.tag_parallel_level(n);

    bn1_alpha.then(bn1_beta, ffout)
             .then(bn2_alpha, computation::root)
             .then(bn2_beta, ffout)
             .then(zero_conv1, computation::root)
             .then(conv1_init, n)
             .then(conv1, x_b)
             .then(bn1, x_b)
             .then(relu1, xx)
             .then(conv1_init_conclude, y)
             .then(conv1_conclude, y)
             .then(bn1_conclude, y)
             .then(relu1_conclude, x_conclude)
             .then(conv2_init, n)
             .then(conv2, x_b)
             .then(bn2, x_b)
             .then(conv2_init_conclude, y)
             .then(conv2_conclude, y)
             .then(bn2_conclude, y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv1_buf("conv1_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N + 2, N + 2, FOUT_BLOCKING}, p_float32, a_input);
    buffer conv2_buf("conv2_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

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
    bn1.store_in(&conv1_buf, {n, fout_b, y + 1, x_bound + 1, ffout});
    relu1.store_in(&conv1_buf, {n, fout_b, y + 1, x_bound + 1, ffout});

    conv1_init_conclude.store_in(&reg1_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv1_conclude.store_in(&reg1_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    bn1_conclude.store_in(&conv1_buf, {n, fout_b, y + 1, x_conclude + 1, ffout});
    relu1_conclude.store_in(&conv1_buf, {n, fout_b, y + 1, x_conclude + 1, ffout});

    /*
     * Second convolution storage
     */
    relu1_padded.store_in(&conv1_buf);

    conv2_init.store_in(&reg2_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    conv2.store_in(&reg2_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    bn2.store_in(&conv2_buf, {n, fout_b, y, x_bound, ffout});

    conv2_init_conclude.store_in(&reg2_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv2_conclude.store_in(&reg2_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    bn2_conclude.store_in(&conv2_buf, {n, fout_b, y, x_conclude, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(),
        filter1.get_buffer(),
        bias1.get_buffer(),
        bn1_scale.get_buffer(),
        bn1_shift.get_buffer(),
        bn1_mean.get_buffer(),
        bn1_variance.get_buffer(),
        filter2.get_buffer(),
        bias2.get_buffer(),
        bn2_scale.get_buffer(),
        bn2_shift.get_buffer(),
        bn2_mean.get_buffer(),
        bn2_variance.get_buffer(),
        &conv1_buf,
        &conv2_buf
    }, "fused_resnet_block_generator_tiramisu.o");

    return 0;
}
