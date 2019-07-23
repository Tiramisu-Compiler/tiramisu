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
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y), fout("fout", 0, FOut);

    var fin_b("fin_b", 0, FIN_NB_BLOCKS), ffin("ffin", 0, FIN_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, fin_b, y, x, ffin}, p_float32);
    input conv_filter("conv_filter", {fout_b, fin_b, k_y, k_x, ffin, ffout}, p_float32);
    input conv_bias("conv_bias", {fout_b, ffout}, p_float32);

    // Pad input
    var x_pad("x", 0, N + 2), y_pad("y", 0, N + 2);

    computation init_input_padded("init_input_padded", {n, fin_b, y_pad, x_pad, ffin}, cast(p_float32, 0));
    computation copy_input("copy_input", {n, fin_b, y, x, ffin}, c_input(n, fin_b, y, x, ffin));
    view input_padded("input_padded", {n, fin_b, y_pad, x_pad, ffin}, p_float32);

    // Init output with zeros
    var x_out("x_out", 0, N/2), y_out("y_out", 0, N/2);
    computation init_output("init_output", {n, fout_b, y_out, x_out, ffout}, cast(p_float32, 0));

    // Convolution computation
    computation init_conv_output("init_conv_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, y, x, fin_b, k_y, k_x, ffin, ffout}, 
        init_conv_output(n, fout_b, y, x, ffout) + input_padded(n, fin_b, y + k_y, x + k_x, ffin)*conv_filter(fout_b, fin_b, k_y, k_x, ffin, ffout)
    );

    // MaxPool computation
    view c_output("c_output", {n, fout_b, y, x, ffout}, p_float32);

    computation maxpool(
        "maxpool",
        {n, fout_b, y, x, ffout},
        expr(
            o_max,
            c_output(n, fout_b, y, x, ffout),
            conv(n, fout_b, y, x, FIN_NB_BLOCKS - 1, K_Y - 1, K_X - 1, FIN_BLOCKING - 1, ffout)
        )
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    init_input_padded.then(copy_input, n)
                     .then(init_output, n)
                     .then(init_conv_output, n)
                     .then(conv, y)
                     .then(maxpool, y);
                     
    conv.interchange(x, fin_b);
    conv.interchange(x, k_y);
    conv.interchange(x, k_x);
    
    conv.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    conv.tag_parallel_level(n);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer input_padded_buf("input_padded_buf", {BATCH_SIZE, FIN_NB_BLOCKS, N + 2, N + 2, FIN_BLOCKING}, p_float32, a_temporary);
    buffer workspace_buf("workspace_buf", {BATCH_SIZE, N, FOUT_BLOCKING}, p_float32, a_temporary);
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    init_input_padded.store_in(&input_padded_buf);
    copy_input.store_in(&input_padded_buf, {n, fin_b, y + 1, x + 1, ffin});
    input_padded.store_in(&input_padded_buf);

    init_output.store_in(&output_buf);

    init_conv_output.store_in(&workspace_buf, {n, x, ffout});
    conv.store_in(&workspace_buf, {n, x, ffout});

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

