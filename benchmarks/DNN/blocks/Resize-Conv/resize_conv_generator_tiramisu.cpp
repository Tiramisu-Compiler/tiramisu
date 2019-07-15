#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

expr mixf(expr x, expr y, expr a)
{
    return x * (1 - a) + y * a;
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
    computation init_resized_input("init_resized_input", {n, y_pad, x_pad, fin}, 0.f);

    expr o_r((cast(p_float32, y) + 0.5f) * (cast(p_float32, IMG_HEIGHT) / cast(p_float32, N)) - 0.5f);
    expr o_c((cast(p_float32, x) + 0.5f) * (cast(p_float32, IMG_WIDTH) / cast(p_float32, N)) - 0.5f);

    expr r_coeff(expr(o_r) - expr(o_floor, o_r));
    expr c_coeff(expr(o_c) - expr(o_floor, o_c));

    expr A00_r(cast(p_int32, expr(o_floor, o_r)));
    expr A00_c(cast(p_int32, expr(o_floor, o_c)));

    computation resize(
        "resize",
        {n, y, x, fin},
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

    view input_resized("input_resized", {n, y_pad, x_pad, fin}, p_float32);

    // Convolution computation
    computation init_output("init_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, y, x, k_y, k_x, fin, ffout}, 
        init_output(n, fout_b, y, x, ffout) + input_resized(n, y + k_y, x + k_x, fin)*conv_filter(fout_b, k_y, k_x, fin, ffout)
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    init_resized_input.then(resize, computation::root)
                      .then(init_output, computation::root)
                      .then(conv, computation::root);

    resize.tag_parallel_level(n);
    resize.vectorize(x, 8);
    resize.tag_unroll_level(fin);

    //n, fout_b, y, x, k_y, k_x, fin, ffout
    conv.interchange(x, k_y);
    conv.interchange(x, k_x);
    //n, fout_b, y, k_y, k_x, x, fin, ffout
    
    conv.tag_parallel_level(n);
    conv.vectorize(ffout, FOUT_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer input_resized_buf("input_resized_buf", {BATCH_SIZE, N + 2, N + 2, FIn}, p_float32, a_temporary);
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

    init_resized_input.store_in(&input_resized_buf);
    resize.store_in(&input_resized_buf, {n, y + 1, x + 1, fin});
    input_resized.store_in(&input_resized_buf);

    init_output.store_in(&output_buf);
    conv.store_in(&output_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(), 
        &output_buf
    }, "resize_conv_tiramisu.o");

    return 0;
}
