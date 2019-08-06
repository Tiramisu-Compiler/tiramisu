/*
Pseudo-code for our convolution :

FOUT_NB_BLOCKS = FOut/FOUT_BLOCKING
X_NB_BLOCKS = N/X_BLOCKING

*_BLOCKING : blocking factor for the given dimension (see configure.h).
regs : buffer of size {X_BLOCKING, FOUT_BLOCKING}.
       We rely on the compiler to map this buffer to CPU vector registers.

for (n = 0; n < BATCH_SIZE; ++n)
for (fout_b = 0; fout_b < FOUT_NB_BLOCKS; ++fout_b)
for (y = 0; y < N; ++y)
for (x_b = 0; x_b < X_NB_BLOCKS; ++x_b)
    for (xx = 0; xx < X_BLOCKING; ++xx)
        regs[xx, :] = bias[fout_b, :]

    for (k_y = 0; k_y < K; ++k_y)
    for (k_x = 0; k_x < K; ++k_x)
    for (fin = 0; fin < FIn; ++fin)
    for (xx = 0; xx < X_BLOCKING; ++xx)
        regs[xx, :] += filter[fout_b, k_y, k_x, fin, :] * input[n, y + k_y, x*X_BLOCKING + xx + k_x, fin]

    for (xx = 0; xx < X_BLOCKING; ++x)
        output[n, fout_b, y, x*X_BLOCKING + xx, :] = regs[xx, :]
*/

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
    var x_b("x_b", 0, X_NB_BLOCKS), xx;

    // Loop through weights to load them into cache
    computation prefetch_weights(
        "prefetch_weights",
        {n, fout_b, y, x_b, k_y, k_x, fin, ffout},
        filter(fout_b, k_y, k_x, fin, ffout)
    );

    // This computation is here to apply register blocking.
    // Convolution intermediate results will be stored in a small buffer that
    // will be mapped to CPU registers (more precisely, CPU vector registers) 
    // instead of being mapped to memory.
    // This computation moves data from our small buffer to the output buffer
    computation reg_store(
        "reg_store",
        {n, fout_b, y, x, ffout},
        conv(n, fout_b, y, x, 0, 0, 0, ffout)
    );
    
    // We split computations over dimension x to apply register blocking
    conv_init.split(x, X_BLOCKING, x_b, xx);
    conv.split(x, X_BLOCKING, x_b, xx);
    reg_store.split(x, X_BLOCKING, x_b, xx);
    
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
    reg_store.vectorize(ffout, FOUT_BLOCKING);
    
    // Note that reg_store is scheduled after that convolution intermediate results are computed
    conv_init.then(prefetch_weights, x_b)
             .then(conv, x_b)
             .then(reg_store, x_b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv_buf("conv_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);
    
    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    prefetch_weights.store_in(&prefetch_w_buf, {});

    // Convolution intermediate results are stored in reg_buf.
    conv_init.store_in(&reg_buf, {x%X_BLOCKING, ffout});
    conv.store_in(&reg_buf, {x%X_BLOCKING, ffout});

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
