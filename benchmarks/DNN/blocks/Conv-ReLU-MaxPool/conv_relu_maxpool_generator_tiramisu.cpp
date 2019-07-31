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
    computation init_conv_output("init_conv_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b, ffout));
    computation conv(
        "conv", 
        {n, fout_b, y, x, k_y, k_x, fin, ffout}, 
        init_conv_output(n, fout_b, y, x, ffout) + c_input(n, y + k_y, x + k_x, fin)*conv_filter(fout_b, k_y, k_x, fin, ffout)
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
        
        init_conv_output.tile(y, x, Y_BLOCKING, X_BLOCKING);
        conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        maxpool.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        
        // n, fout_b, y_b, x_b, yy, xx, k_y, k_x, fin, ffout
        conv.interchange(xx, k_y);
        conv.interchange(xx, k_x);
        // n, fout_b, y_b, x_b, yy, k_y, k_x, xx, fin, ffout
        conv.interchange(yy, k_y);
        conv.interchange(yy, k_x);
        // n, fout_b, y_b, x_b, k_y, k_x, yy, xx, fin, ffout
        
        init_output.then(init_conv_output, fout_b)
                   .then(prefetch_weights, x_b)
                   .then(conv, x_b)
                   .then(maxpool, x_b);
                   
        conv.tag_parallel_level(fout_b);
    }
    
    else {
        // n, fout_b, y, x, k_y, k_x, fin, ffout
        conv.interchange(x, k_y);
        
        var xx;
        conv.split(x, X_BLOCKING, x_b, xx);
        conv.interchange(xx, k_x);
        // n, fout_b, y, k_y, x_b, k_x, xx, fin, ffout
        
        init_output.then(init_conv_output, fout_b)
                   .then(conv, y)
                   .then(maxpool, y);
    }
    
    conv.tag_parallel_level(n);
    
    init_conv_output.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    // Workspace buffer is used to store result of convolution
    std::vector<expr> wb_dims = {BATCH_SIZE, FOUT_NB_BLOCKS, Y_BLOCKING, X_BLOCKING, FOUT_BLOCKING};
    if (N < 224)
        wb_dims = {BATCH_SIZE, N, FOUT_BLOCKING};
        
    buffer workspace_buf("workspace_buf", wb_dims, p_float32, a_input);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    if (SCHEDULE_PREFETCH_WEIGHTS)
        prefetch_weights.store_in(&prefetch_w_buf, {});

    init_output.store_in(&output_buf);

    if (N >= 224) {
        init_conv_output.store_in(&workspace_buf, {n, fout_b, y%Y_BLOCKING, x%X_BLOCKING, ffout});
        conv.store_in(&workspace_buf, {n, fout_b, y%Y_BLOCKING, x%X_BLOCKING, ffout});
    }
    
    else {
        init_conv_output.store_in(&workspace_buf, {n, x, ffout});
        conv.store_in(&workspace_buf, {n, x, ffout});
    }

    c_output.store_in(&output_buf, {n, fout_b, y/2, x/2, ffout});
    maxpool.store_in(&output_buf, {n, fout_b, y/2, x/2, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(),
        &workspace_buf,
        &output_buf
    }, "conv_relu_maxpool_tiramisu.o");

    return 0;
}

