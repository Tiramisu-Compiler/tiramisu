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

    // Convolution computation
    computation init_conv_output("init_conv_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, fin_b, y, x, k_y, k_x, ffin, ffout}, 
        init_conv_output(n, fout_b, y, x, ffout) + input_padded(n, fin_b, y + k_y, x + k_x, ffin)*conv_filter(fout_b, fin_b, k_y, k_x, ffin, ffout)
    );

    // MaxPool computation
    var x_out("x_out", 0, N/2), y_out("y_out", 0, N/2);
    var p_x("p_x", 0, 2), p_y("p_y", 0, 2);

    computation init_output("init_output", {n, fout_b, y_out, x_out, ffout}, cast(p_float32, 0));
    view c_output("c_output", {n, fout_b, y_out, x_out, ffout}, p_float32);

    computation maxpool(
        "maxpool",
        {n, fout_b, y_out, x_out, p_y, p_x, ffout},
        expr(
            o_max,
            c_output(n, fout_b, y_out, x_out, ffout),
            conv(n, fout_b, FIN_NB_BLOCKS - 1, y_out*2 + p_y, x_out*2 + p_x, K_Y - 1, K_X - 1, FIN_BLOCKING - 1, ffout)
        )
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    init_input_padded.then(copy_input, computation::root)
                     .then(init_conv_output, computation::root)
                     .then(conv, fout_b)
                     .then(init_output, computation::root)
                     .then(maxpool, x_out);

    init_input_padded.tag_parallel_level(n);

    copy_input.tag_parallel_level(n);
    copy_input.vectorize(ffin, FIN_BLOCKING);

    //n, fout_b, fin_b, y, x, k_y, k_x, ffin, ffout
    conv.interchange(x, k_y);
    conv.interchange(x, k_x);
    //n, fout_b, fin_b, y, k_y, k_x, x, ffin, ffout
    
    conv.tag_parallel_level(n);
    conv.vectorize(ffout, FOUT_BLOCKING);

    maxpool.tag_parallel_level(n);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer input_padded_buf("input_padded_buf", {BATCH_SIZE, FIN_NB_BLOCKS, N + 2, N + 2, FIN_BLOCKING}, p_float32, a_temporary);
    buffer workspace_buf("workspace_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_temporary);
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    init_input_padded.store_in(&input_padded_buf);
    copy_input.store_in(&input_padded_buf, {n, fin_b, y + 1, x + 1, ffin});
    input_padded.store_in(&input_padded_buf);

    init_conv_output.store_in(&workspace_buf);
    conv.store_in(&workspace_buf, {n, fout_b, y, x, ffout});

    init_output.store_in(&output_buf);
    c_output.store_in(&output_buf);
    maxpool.store_in(&output_buf, {n, fout_b, y_out, x_out, ffout});

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
