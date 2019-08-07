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

    // This computation is here to apply register blocking.
    // Convolution intermediate results will be stored in a small buffer that
    // will be mapped to CPU registers (more precisely, CPU vector registers) 
    // instead of being mapped to memory.
    // This computation moves data from our small buffer to the output buffer
    computation reg_store(
        "reg_store",
        {n, fout_b, y, x, ffout},
        conv(n, fout_b, y, x, 0, 0, 0, 0, ffout)
    );
    
    // We split computations over dimension x to apply register blocking
    var x_b, y_b;
    var xx, yy;

    conv_init.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
    conv.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
    reg_store.tile(y, x, Y_BLOCKING, X_BLOCKING, y_b, x_b, yy, xx);
    
    // n, fout_b, y, x_b, xx, fin_b, k_y, k_x, fin, ffout
    conv.interchange(xx, fin_b);
    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    conv.interchange(xx, ffin);
    conv.interchange(xx, ffout);

    conv.interchange(yy, fin_b);
    conv.interchange(yy, k_y);
    conv.interchange(yy, k_x);
    conv.interchange(yy, ffin);
    conv.interchange(yy, ffout);
    // n, fout_b, y, x_b, fin_b, k_y, k_x, fin, ffout, xx

    conv.tag_unroll_level(k_y);
    conv.tag_parallel_level(fout_b);
    conv.tag_parallel_level(n);
    
    conv_init.vectorize(ffout, FOUT_BLOCKING);
    conv.vectorize(ffout, FOUT_BLOCKING);
    reg_store.vectorize(ffout, FOUT_BLOCKING);
    
    // Note that reg_store is scheduled after that convolution intermediate results are computed
    conv_init.then(conv, x_b)
             .then(reg_store, x_b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv_buf("conv_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);
    
    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {Y_BLOCKING, X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    // Convolution intermediate results are stored in reg_buf.
    conv_init.store_in(&reg_buf, {y%Y_BLOCKING, x%X_BLOCKING, ffout});
    conv.store_in(&reg_buf, {y%Y_BLOCKING, x%X_BLOCKING, ffout});

    // reg_store computation moves data from reg_buf to conv_buf.
    reg_store.store_in(&conv_buf, {n, fout_b, y, x, ffout});

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
