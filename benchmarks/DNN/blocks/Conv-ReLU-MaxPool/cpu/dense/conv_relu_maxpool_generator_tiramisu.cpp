#define __TIRAMISU_GENERATOR__
#include <tiramisu/tiramisu.h>
#include <vector>
#include "configure.h"

using namespace tiramisu;

int main()
{
    init("conv_relu_maxpool_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), fin("fin", 0, FIn), n("n", 0, BATCH_SIZE);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y);

    var x_pad("x", 0, N + 2), y_pad("y", 0, N + 2);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, y_pad, x_pad, fin}, p_float32);
    input conv_filter("conv_filter", {fout_b, k_y, k_x, fin, ffout}, p_float32);
    input conv_bias("conv_bias", {fout_b, ffout}, p_float32);

    // Init output with zeros
    // With this computation, we can avoid to compute ReLU
    var x_out("x_out", 0, N/2), y_out("y_out", 0, N/2);
    computation init_output("init_output", {n, fout_b, y_out, x_out, ffout}, cast(p_float32, 0));

    // Convolution computation
    computation conv_init("conv_init", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv",
        {n, fout_b, y, x, k_y, k_x, fin, ffout},
        conv_init(n, fout_b, y, x, ffout) + c_input(n, y + k_y, x + k_x, fin)*conv_filter(fout_b, k_y, k_x, fin, ffout)
    );

    // MaxPool computation
    view c_output("c_output", {n, fout_b, y, x, ffout}, p_float32);

    computation maxpool(
        "maxpool",
        {n, fout_b, y, x, ffout},
        expr(
            o_max,
            c_output(n, fout_b, y, x, ffout),
            conv(n, fout_b, y, x, 0, 0, 0, ffout)
        )
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var x_b("x_b", 0, X_NB_BLOCKS), xx;

    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fout_b, y, x_b, k_y, k_x, fin, ffout},
        conv_filter(fout_b, k_y, k_x, fin, ffout)
    );

    // We split computations over dimension x to apply register blocking
    conv_init.split(x, X_BLOCKING, x_b, xx);
    conv.split(x, X_BLOCKING, x_b, xx);
    maxpool.split(x, X_BLOCKING, x_b, xx);

    // n, fout_b, y, x_b, xx, k_y, k_x, fin, ffout
    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    conv.interchange(xx, fin);
    conv.interchange(xx, ffout);
    // n, fout_b, y, x_b, k_y, k_x, fin, ffout, xx

    conv.tag_parallel_level(fout_b);
    conv.tag_parallel_level(n);

    conv_init.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    init_output.then(conv_init, fout_b)
               .then(prefetch_weights, x_b)
               .then(conv, x_b)
               .then(maxpool, x_b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    prefetch_weights.store_in(&prefetch_w_buf, {});

    init_output.store_in(&output_buf);

    conv_init.store_in(&reg_buf, {x%X_BLOCKING, ffout});
    conv.store_in(&reg_buf, {x%X_BLOCKING, ffout});

    c_output.store_in(&output_buf, {n, fout_b, y/2, x/2, ffout});
    maxpool.store_in(&output_buf, {n, fout_b, y/2, x/2, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        conv_filter.get_buffer(),
        conv_bias.get_buffer(),
        &output_buf
    }, "conv_relu_maxpool_tiramisu.o");

    return 0;
}
