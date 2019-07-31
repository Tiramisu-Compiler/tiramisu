#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

expr mixf(expr x, expr y, expr a)
{
    return x + (y - x) * a;
}

int main()
{
    init("resize_conv_block");

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
    computation init_output("init_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, y, x, k_y, k_x, fin, ffout}, 
        init_output(n, fout_b, y, x, ffout) + conv_filter(fout_b, k_y, k_x, fin, ffout)*resized_view(n, y + k_y, x + k_x, fin)
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var y_b("y_b", 0, Y_NB_BLOCKS), x_b("x_b", 0, X_NB_BLOCKS);
    
    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fout_b, y_b, x_b, k_y, k_x, fin, ffout},
        conv_filter(fout_b, k_y, k_x, fin, ffout),
        SCHEDULE_PREFETCH_WEIGHTS
    );
    
    if (N >= 224) {
        var xx, yy;
        
        resize.split(y, Y_BLOCKING, y_b, yy);
        init_output.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        
        // n, fout_b, y_b, x_b, yy, xx, k_y, k_x, fin, ffout
        conv.interchange(xx, k_y);
        conv.interchange(xx, k_x);
        // n, fout_b, y_b, x_b, yy, k_y, k_x, xx, fin, ffout
        conv.interchange(yy, k_y);
        conv.interchange(yy, k_x);
        // n, fout_b, y_b, x_b, k_y, k_x, yy, xx, fin, ffout

        // Interchange fout_b and y_b in order to fuse resize with convolution
        prefetch_weights.interchange(fout_b, y_b);
        init_output.interchange(fout_b, y_b);
        conv.interchange(fout_b, y_b);
        
        resize_prelude.then(resize, n)
                      .then(init_output, y_b)
                      .then(prefetch_weights, x_b)
                      .then(conv, x_b);
    }
    
    else {
        // n, fout_b, y, x, k_y, k_x, fin, ffout
        conv.interchange(x, k_y);
        
        var xx;
        conv.split(x, X_BLOCKING, x_b, xx);
        conv.interchange(xx, k_x);
        // n, fout_b, y, k_y, x_b, k_x, xx, fin, ffout

        // Interchange fout_b and y in order to fuse resize with convolution
        init_output.interchange(fout_b, y);
        conv.interchange(fout_b, y);
        
        resize_prelude.then(resize, n)
                      .then(init_output, y)
                      .then(conv, fout_b);
    }
    
    conv.tag_parallel_level(n);

    resize.tag_unroll_level(fin);
    resize.vectorize(x_pad, 8);
    
    init_output.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer resized_buf("input_resized_buf", {BATCH_SIZE, N + 2, N + 2, FIn}, p_float32, a_input);
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    if (SCHEDULE_PREFETCH_WEIGHTS)
        prefetch_weights.store_in(&prefetch_w_buf, {});

    resize_prelude.store_in(&resized_buf, {n, y_prelude, x_pad, fin});
    resize.store_in(&resized_buf, {n, y + 2, x_pad, fin});
    resized_view.store_in(&resized_buf);

    init_output.store_in(&output_buf);
    conv.store_in(&output_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(),
        &resized_buf,
        &output_buf
    }, "resize_conv_tiramisu.o");

    return 0;
}
