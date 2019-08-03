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

    var fin_b("fin_b", 0, FIN_NB_BLOCKS), ffin("ffin", 0, FIN_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);
    
    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);

    input c_input("c_input", {n, fin_b, y_pad, x_pad, ffin}, p_float32);
    input filter("filter", {fout_b, fin_b, k_y, k_x, ffin, ffout}, p_float32);
    input bias("bias", {fout_b, ffout}, p_float32);

    computation conv_init("conv_init", {n, fout_b, y, x, ffout}, bias(fout_b, ffout));
    
    computation conv(
        "conv",
        {n, fout_b, y, x, fin_b, k_y, k_x, ffin, ffout},
        conv_init(n, fout_b, y, x, ffout) + filter(fout_b, fin_b, k_y, k_x, ffin, ffout) * c_input(n, fin_b, y + k_y, x + k_x, ffin)
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    var y_b("y_b", 0, Y_NB_BLOCKS), x_b("x_b", 0, X_NB_BLOCKS);
    var yy, xx;
    
    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fout_b, y_b, x_b, fin_b, k_y, k_x, ffin, ffout},
        filter(fout_b, fin_b, k_y, k_x, ffin, ffout),
        SCHEDULE_PREFETCH_WEIGHTS
    );
        
    conv_init.tile(y, x, Y_BLOCKING, X_BLOCKING);
    conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
        
    // n, fout_b, y_b, x_b, yy, xx, fin_b, k_y, k_x, ffin, ffout
    conv.interchange(xx, fin_b);
    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    // n, fout_b, y_b, x_b, yy, fin_b, k_y, k_x, xx, ffin, ffout
    conv.interchange(yy, fin_b);
    conv.interchange(yy, k_y);
    conv.interchange(yy, k_x);
    // n, fout_b, y_b, x_b, fin_b, k_y, k_x, yy, xx, ffin, ffout
    
    conv.tag_parallel_level(fout_b);
    conv.tag_parallel_level(n);
    
    conv_init.vectorize(ffout, VEC_LEN);
    conv.vectorize(ffout, VEC_LEN);

    if (SCHEDULE_PREFETCH_WEIGHTS)
        conv_init.then(prefetch_weights, x_b)
                 .then(conv, fin_b);
                 
    else
        conv_init.then(conv, x_b);


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
