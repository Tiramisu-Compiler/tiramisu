#define __TIRAMISU_GENERATOR__
#include <tiramisu/tiramisu.h>
#include "configure.h"
#define SCHEDULE_CPU 1

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("vgg_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), n("n", 0, BATCH_SIZE);
    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);

    var o_x("o_x", 0, N/2), o_y("o_y", 0, N/2);
    var k_x("k_x", 0, K), k_y("k_y", 0, K);

    var fin1("fin1", 0, FIn);
    var fin2_b("fin2_b", 0, FIN2_NB_BLOCKS), ffin2("ffin2", 0, FIN2_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    // Input computations
    input c_input("c_input", {n, y_pad, x_pad, fin1}, p_float32);

    input bias1("bias1", {fin2_b, ffin2}, p_float32);
    input filter1("filter1", {fin2_b, k_y, k_x, fin1, ffin2}, p_float32);

    input bias2("bias2", {fout_b, ffout}, p_float32);
    input filter2("filter2", {fout_b, fin2_b, k_y, k_x, ffin2, ffout}, p_float32);

    // First conv computations
    computation zero_conv1("zero_conv1", {n, fin2_b, y_pad, x_pad, ffin2}, cast(p_float32, 0));

    computation conv1_init("conv1_init", {n, fin2_b, y, x, ffin2}, bias1(fin2_b, ffin2));
    computation conv1(
        "conv1",
        {n, fin2_b, y, x, k_y, k_x, fin1, ffin2},
        conv1_init(n, fin2_b, y, x, ffin2) + filter1(fin2_b, k_y, k_x, fin1, ffin2) * c_input(n, y + k_y, x + k_x, fin1)
    );

    // First relu
    computation relu1(
        "relu1",
        {n, fin2_b, y, x, ffin2},
        expr(
            o_max,
            0.f,
            conv1(n, fin2_b, y, x, 0, 0, 0, ffin2)
        )
    );

    view relu1_padded("relu1_padded", {n, fin2_b, y_pad, x_pad, ffin2}, p_float32);

    // Second conv computations

    // x_bound is used to have the width dimension divisible by X_BLOCKING
    // in the conv computation.
    var x2_bound("x2_bound", 0, X2_BOUND);
    var x2_conclude("x2_conclude", X2_BOUND, N);

    computation conv2_init("conv2_init", {n, y, x2_bound, fout_b, ffout}, bias2(fout_b, ffout));
    computation conv2(
        "conv2",
        {n, y, x2_bound, fin2_b, k_y, k_x, ffin2, fout_b, ffout},
        conv2_init(n, y, x2_bound, fout_b, ffout) + filter2(fout_b, fin2_b, k_y, k_x, ffin2, ffout) * relu1_padded(n, fin2_b, y + k_y, x2_bound + k_x, ffin2)
    );

    computation conv2_init_conclude("conv2_init_conclude", {n, y, fout_b, ffout, x2_conclude}, bias2(fout_b, ffout));
    computation conv2_conclude(
        "conv2_conclude",
        {n, y, fin2_b, k_y, k_x, ffin2, fout_b, ffout, x2_conclude},
        conv2_init_conclude(n, y, fout_b, ffout, x2_conclude) + filter2(fout_b, fin2_b, k_y, k_x, ffin2, ffout) * relu1_padded(n, fin2_b, y + k_y, x2_conclude + k_x, ffin2)
    );

    // Maxpooling computation
    computation maxpool_init("maxpool_init", {n, fout_b, o_y, o_x, ffout}, cast(p_float32, 0.f));
    view c_output("c_output", {n, fout_b, y, x, ffout}, p_float32);

    computation maxpool(
        "maxpool",
        {n, y, x2_bound, fout_b, ffout},
        expr(
            o_max,
            c_output(n, fout_b, y, x2_bound, ffout),
            conv2(n, y, x2_bound, 0, 0, 0, 0, fout_b, ffout)
        )
    );

    view c_output_conclude("c_output_conclude", {n, fout_b, y, x2_conclude, ffout}, p_float32);
    computation maxpool_conclude(
        "maxpool_conclude",
        {n, y, fout_b, ffout, x2_conclude},
        expr(
            o_max,
            c_output_conclude(n, fout_b, y, x2_conclude, ffout),
            conv2_conclude(n, y, 0, 0, 0, 0, fout_b, ffout, x2_conclude)
        )
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var x1_b("x1_b", 0, X1_NB_BLOCKS), xx1;

    /*
     * Schedule for first Conv-ReLU
     */
    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fin2_b, y, x1_b, k_y, k_x, fin1, ffin2},
        filter1(fin2_b, k_y, k_x, fin1, ffin2)
    );

    // We split computations over dimension x to apply register blocking
    conv1_init.split(x, X1_BLOCKING, x1_b, xx1);
    conv1.split(x, X1_BLOCKING, x1_b, xx1);
    relu1.split(x, X1_BLOCKING, x1_b, xx1);

    // n, fin2_b, y, x_b, xx, k_y, k_x, fin1, ffin2
    conv1.interchange(xx1, k_y);
    conv1.interchange(xx1, k_x);
    conv1.interchange(xx1, fin1);
    conv1.interchange(xx1, ffin2);
    // n, fin2_b, y, x_b, k_y, k_x, fin1, ffin2, xx

    conv1_init.vectorize(ffin2, VEC_LEN);
    conv1.vectorize(ffin2, VEC_LEN);
    relu1.vectorize(ffin2, VEC_LEN);

    /*
     * Schedule for second Conv-ReLU-MaxPool
     */
    // Split over dimension x
    var x2_b, xx2;
    conv2.split(x2_bound, X2_BLOCKING, x2_b, xx2);

    conv2.interchange(xx2, fin2_b);
    conv2.interchange(xx2, k_y);
    conv2.interchange(xx2, k_x);
    conv2.interchange(xx2, ffin2);
    conv2.interchange(xx2, fout_b);
    conv2.interchange(xx2, ffout);

    conv2_init.split(x2_bound, X2_BLOCKING, x2_b, xx2);
    maxpool.split(x2_bound, X2_BLOCKING, x2_b, xx2);

    conv2_init.interchange(xx2, fout_b);
    conv2_init.interchange(xx2, ffout);

    maxpool.interchange(xx2, fout_b);
    maxpool.interchange(xx2, ffout);

    // Vectorize and unroll
    conv2_init.vectorize(ffout, FOUT_BLOCKING);
    conv2.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    conv2.tag_unroll_level(xx2);
    conv2.tag_unroll_level(fout_b);

    conv2_init.tag_unroll_level(xx2);
    conv2_init.tag_unroll_level(fout_b);

    maxpool.tag_unroll_level(xx2);
    maxpool.tag_unroll_level(fout_b);

    // schedule for conv_conclude
    // This schedule is the same as conv computation
    conv2_init_conclude.vectorize(ffout, FOUT_BLOCKING);
    conv2_conclude.vectorize(ffout, FOUT_BLOCKING);
    maxpool_conclude.vectorize(ffout, FOUT_BLOCKING);

    conv2_conclude.tag_unroll_level(x2_conclude);
    conv2_conclude.tag_unroll_level(fout_b);

    conv2_init_conclude.tag_unroll_level(x2_conclude);
    conv2_init_conclude.tag_unroll_level(fout_b);

    maxpool_conclude.tag_unroll_level(x2_conclude);
    maxpool_conclude.tag_unroll_level(fout_b);

    // Parallelize and order
    conv2.tag_parallel_level(n);

    maxpool_init.then(zero_conv1, fin2_b)
                .then(conv1_init, fin2_b)
                .then(prefetch_weights, x1_b)
                .then(conv1, x1_b)
                .then(relu1, x1_b)
                .then(conv2_init, n)
                .then(conv2, x2_b)
                .then(maxpool, x2_b)
                .then(conv2_init_conclude, y)
                .then(conv2_conclude, y)
                .then(maxpool_conclude, y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv1_buf("conv1_buf", {BATCH_SIZE, FIN2_NB_BLOCKS, N + 2, N + 2, FIN2_BLOCKING}, p_float32, a_input);
    buffer maxpool_buf("maxpool_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg1_buf("reg1_buf", {X1_BLOCKING, FIN2_BLOCKING}, p_float32, a_temporary);
    buffer reg2_buf("reg2_buf", {FOUT_NB_BLOCKS, X2_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    prefetch_weights.store_in(&prefetch_w_buf, {});

    /*
     * Storage for first conv
     */
    zero_conv1.store_in(&conv1_buf);
    conv1_init.store_in(&reg1_buf, {x%X1_BLOCKING, ffin2});
    conv1.store_in(&reg1_buf, {x%X1_BLOCKING, ffin2});
    relu1.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});

    relu1_padded.store_in(&conv1_buf);

    /*
     * Storage for second conv
     */
    conv2_init.store_in(&reg2_buf, {fout_b, x2_bound%X2_BLOCKING, ffout});
    conv2.store_in(&reg2_buf, {fout_b, x2_bound%X2_BLOCKING, ffout});
    maxpool.store_in(&maxpool_buf, {n, fout_b, y/2, x2_bound/2, ffout});

    conv2_init_conclude.store_in(&reg2_buf, {fout_b, x2_conclude%X2_BLOCKING, ffout});
    conv2_conclude.store_in(&reg2_buf, {fout_b, x2_conclude%X2_BLOCKING, ffout});

    if (N % X2_BLOCKING > 1) {
        c_output_conclude.store_in(&maxpool_buf, {n, fout_b, y/2, x2_conclude/2, ffout});
        maxpool_conclude.store_in(&maxpool_buf, {n, fout_b, y/2, x2_conclude/2, ffout});
    }

    else {
        c_output_conclude.store_in(&maxpool_buf, {n, fout_b, y/2, (N-1)/2, ffout});
        maxpool_conclude.store_in(&maxpool_buf, {n, fout_b, y/2, (N-1)/2, ffout});
    }

    maxpool_init.store_in(&maxpool_buf);
    c_output.store_in(&maxpool_buf, {n, fout_b, y/2, x/2, ffout});

    // -------------------------------------------------------
    // Code generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(),
        filter1.get_buffer(),
        bias1.get_buffer(),
        filter2.get_buffer(),
        bias2.get_buffer(),
        &conv1_buf,
        &maxpool_buf
    }, "generated_vgg_block.o");

    return 0;
}
