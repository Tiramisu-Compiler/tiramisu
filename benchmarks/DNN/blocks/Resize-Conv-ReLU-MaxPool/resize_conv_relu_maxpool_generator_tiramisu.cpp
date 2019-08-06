#include <tiramisu/tiramisu.h>
#include <vector>
#include "configure.h"

using namespace tiramisu;

expr mixf(expr x, expr y, expr a)
{
    return x + (y - x) * a;
}

int main()
{
    init("resize_conv_relu_maxpool_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var o_x("o_x", 0, IMG_WIDTH), o_y("o_y", 0, IMG_HEIGHT), fin("fin", 0, FIn), n("n", 0, BATCH_SIZE);
    var x("x", 0, N), y("y", 0, N);

    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y), fout("fout", 0, FOut);

    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, o_y, o_x, fin}, p_float32);
    input conv_filter("conv_filter", {fout_b, k_y, k_x, fin, ffout}, p_float32);
    input conv_bias("conv_bias", {fout_b, ffout}, p_float32);

    // Init output with zeros
    // With this computation, we can avoid to compute ReLU
    var x_out("x_out", 0, N/2), y_out("y_out", 0, N/2);
    computation init_output("init_output", {n, fout_b, y_out, x_out, ffout}, cast(p_float32, 0));

    // Resize computation
    // Compute for y = 2, ..., N + 1
    expr o_r((cast(p_float32, y + 2) + 0.5f) * (cast(p_float32, IMG_HEIGHT) / cast(p_float32, N + 2)) - 0.5f);
    expr o_c((cast(p_float32, x_pad) + 0.5f) * (cast(p_float32, IMG_WIDTH) / cast(p_float32, N + 2)) - 0.5f);

    expr r_coeff(expr(o_r) - expr(o_floor, o_r));
    expr c_coeff(expr(o_c) - expr(o_floor, o_c));

    expr A00_r(cast(p_int32, expr(o_floor, o_r)));
    expr A00_c(cast(p_int32, expr(o_floor, o_c)));

    computation resize(
        "resize",
        {n, y, x_pad, fin},
        mixf(
            mixf(
                c_input(n, A00_r, A00_c, fin), 
                c_input(n, A00_r + 1, A00_c, fin), 
                r_coeff
            ),

            mixf(
                c_input(n, A00_r, A00_c + 1, fin), 
                c_input(n, A00_r + 1, A00_c + 1, fin), 
                r_coeff
            ),
    
            c_coeff
        )
    );

    // Start to compute resize for y = 0, 1 to fuse resize with convolution
    var y_prelude("y_prelude", 0, 2);

    expr o_r_prelude((cast(p_float32, y_prelude) + 0.5f) * (cast(p_float32, IMG_HEIGHT) / cast(p_float32, N + 2)) - 0.5f);
    expr r_coeff_prelude(expr(o_r_prelude) - expr(o_floor, o_r_prelude));
    expr A00_r_prelude(cast(p_int32, expr(o_floor, o_r_prelude)));

    computation resize_prelude(
        "resize_prelude",
        {n, y_prelude, x_pad, fin},
        mixf(
            mixf(
                c_input(n, A00_r_prelude, A00_c, fin), 
                c_input(n, A00_r_prelude + 1, A00_c, fin), 
                r_coeff_prelude
            ),

            mixf(
                c_input(n, A00_r_prelude, A00_c + 1, fin), 
                c_input(n, A00_r_prelude + 1, A00_c + 1, fin), 
                r_coeff_prelude
            ),
    
            c_coeff
        )
    );

    // Convolution computation
    view resized_view("resized_view", {n, y_pad, x_pad, fin}, p_float32);
    computation conv_init("conv_init", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, y, x, k_y, k_x, fin, ffout}, 
        conv_init(n, fout_b, y, x, ffout) + conv_filter(fout_b, k_y, k_x, fin, ffout)*resized_view(n, y + k_y, x + k_x, fin)
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

    // Interchange fout_b and y in order to fuse resize with convolution
    conv_init.interchange(fout_b, y);
    prefetch_weights.interchange(fout_b, y);
    conv.interchange(fout_b, y);
    maxpool.interchange(fout_b, y);

    conv.tag_parallel_level(n);

    resize.vectorize(x_pad, VEC_LEN);    
    conv_init.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    init_output.then(resize_prelude, n)
               .then(resize, n)
               .then(conv_init, y)
               .then(prefetch_weights, x_b)
               .then(conv, x_b)
               .then(maxpool, x_b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer resized_buf("input_resized_buf", {BATCH_SIZE, N + 2, N + 2, FIn}, p_float32, a_input);
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);
    
    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    prefetch_weights.store_in(&prefetch_w_buf, {});

    resize_prelude.store_in(&resized_buf, {n, y_prelude, x_pad, fin});
    resize.store_in(&resized_buf, {n, y + 2, x_pad, fin});
    resized_view.store_in(&resized_buf);

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
        &resized_buf,
        &output_buf
    }, "resize_conv_relu_maxpool_tiramisu.o");

    return 0;
}
