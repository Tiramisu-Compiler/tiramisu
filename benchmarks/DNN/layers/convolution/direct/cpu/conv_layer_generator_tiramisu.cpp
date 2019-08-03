#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("conv_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), n("n", 0, BATCH_SIZE);
    var k_x("k_x", 0, K), k_y("k_y", 0, K);

    var fin("fin", 0, FIn);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);
    
    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);

    input c_input("c_input", {n, y_pad, x_pad, fin}, p_float32);
    input filter("filter", {fout_b, k_y, k_x, fin, ffout}, p_float32);
    input bias("bias", {fout_b, ffout}, p_float32);

    computation conv_init("conv_init", {n, fout_b, y, x, ffout}, bias(fout_b, ffout));
    
    computation conv(
        "conv",
        {n, fout_b, y, x, k_y, k_x, fin, ffout},
        conv_init(n, fout_b, y, x, ffout) + filter(fout_b, k_y, k_x, fin, ffout) * c_input(n, y + k_y, x + k_x, fin)
	);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var y_b("y_b", 0, Y_NB_BLOCKS), x_b("x_b", 0, X_NB_BLOCKS);
    
    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fout_b, y_b, x_b, k_y, k_x, fin, ffout},
        filter(fout_b, k_y, k_x, fin, ffout),
        SCHEDULE_PREFETCH_WEIGHTS
    );
    
    if (N >= 224) {
        var xx, yy;
        
        conv_init.tile(y, x, Y_BLOCKING, X_BLOCKING);
        conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        
        // n, fout_b, y_b, x_b, yy, xx, k_y, k_x, fin, ffout
        conv.interchange(xx, k_y);
        conv.interchange(xx, k_x);
        // n, fout_b, y_b, x_b, yy, k_y, k_x, xx, fin, ffout
        conv.interchange(yy, k_y);
        conv.interchange(yy, k_x);
        // n, fout_b, y_b, x_b, k_y, k_x, yy, xx, fin, ffout
        
        conv_init.then(prefetch_weights, x_b)
                 .then(conv, x_b);
    }
    
    else {
        // n, fout_b, y, x, k_y, k_x, fin, ffout
        conv.interchange(x, k_y);
        
        var xx;
        conv.split(x, X_BLOCKING, x_b, xx);
        conv.interchange(xx, k_x);
        // n, fout_b, y, k_y, x_b, k_x, xx, fin, ffout
        
        conv_init.then(conv, y);
    }
    
    conv.tag_parallel_level(fout_b);
    conv.tag_parallel_level(n);
    
    conv_init.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv_buf("conv_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);
    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);

    if (SCHEDULE_PREFETCH_WEIGHTS)
        prefetch_weights.store_in(&prefetch_w_buf, {});

    conv_init.store_in(&conv_buf);
    conv.store_in(&conv_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(), 
        filter.get_buffer(), 
        bias.get_buffer(), 
        &conv_buf
    },"generated_conv_layer.o");

    return 0;
}
