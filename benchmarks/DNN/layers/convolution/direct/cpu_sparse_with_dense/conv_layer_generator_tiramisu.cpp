#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

#define PATTERN_0 0

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
    input filter2("filter2", {fout_b, fin_b, k_x, ffin, ffout}, p_float32);
    input bias("bias", {fout_b, ffout}, p_float32);

    computation conv_init("conv_init", {n, y, fout_b, x, ffout}, bias(fout_b, ffout));
    view conv_out("conv_out", {n, y, x, fout_b, ffout}, p_float32);
    
    // x_bound is used to have the width dimension divisible by X_BLOCKING
    // in the conv computation.
    var x_bound("x_bound", 0, X_BOUND);
    var x_conclude("x_conclude", X_BOUND, N);

#if PATTERN_0
    expr constraints_P0 =
	((fin_b*FIN_BLOCKING + ffin) >= ZERO_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL) &&
	((fin_b*FIN_BLOCKING + ffin) <  ZERO_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL + PATTERN_0_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL);

    computation conv_P0(
        "conv_P0",
        {n, y, fin_b, x_bound, ffin, fout_b, ffout},
	constraints_P0,
        conv_out(n, y, x_bound, fout_b, ffout) + filter2(fout_b, fin_b, 0, ffin, ffout) * c_input(n, fin_b, y + 0, x_bound + 0, ffin)
					       + filter2(fout_b, fin_b, 1, ffin, ffout) * c_input(n, fin_b, y + 0, x_bound + 1, ffin)
					       + filter2(fout_b, fin_b, 2, ffin, ffout) * c_input(n, fin_b, y + 0, x_bound + 2, ffin));
#endif

    // Compute convolution from 0 to x_bound
    computation conv(
        "conv",
        {n, y, fin_b, x_bound, k_y, k_x, ffin, fout_b, ffout},
	fin_b*8+ffin>COMMON_ZERO_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL,
        conv_out(n, y, x_bound, fout_b, ffout) + filter(fout_b, fin_b, k_y, k_x, ffin, ffout) * c_input(n, fin_b, y + k_y, x_bound + k_x, ffin)
    );

    conv.add_predicate(((fin_b*FIN_BLOCKING + ffin) >= ZERO_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL));

    // Compute convolution from x_bound to N
    computation conv_conclude(
        "conv_conclude",
        {n, y, fin_b, k_y, k_x, ffin, fout_b, ffout, x_conclude},
	(fin_b*FIN_BLOCKING + ffin) >= ZERO_WEIGHT_FILTERS_PER_OUTPUT_CHANNEL,
        conv_out(n, y, x_conclude, fout_b, ffout) + filter(fout_b, fin_b, k_y, k_x, ffin, ffout) * c_input(n, fin_b, y + k_y, x_conclude + k_x, ffin)
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // schedule for conv computation

    // We introduce those two computations to do register blocking
    computation reg_load(
        "reg_load",
        {n, y, fin_b, x_bound, fout_b, ffout},
        conv_init(n, y, fout_b, x_bound, ffout)
    );

    computation reg_store(
        "reg_store",
        {n, y, fin_b, x_bound, fout_b, ffout},
        conv(n, y, fin_b, x_bound, 0, 0, 0, fout_b, ffout)
    );

    // Split over dimension x
    var x_b, xx;
    conv.split(x_bound, X_BLOCKING, x_b, xx);
#if PATTERN_0
    conv_P0.split(x_bound, X_BLOCKING, x_b, xx);
#endif

    conv.interchange(xx, k_y);
    conv.interchange(xx, k_x);
    conv.interchange(xx, ffin);
    conv.interchange(xx, fout_b);
    conv.interchange(xx, ffout);
#if PATTERN_0
    conv_P0.interchange(xx, ffin);
    conv_P0.interchange(xx, fout_b);
    conv_P0.interchange(xx, ffout);
#endif

    reg_load.split(x_bound, X_BLOCKING, x_b, xx);
    reg_store.split(x_bound, X_BLOCKING, x_b, xx);

    reg_load.interchange(xx, fout_b);
    reg_load.interchange(xx, ffout);

    reg_store.interchange(xx, fout_b);
    reg_store.interchange(xx, ffout);

    // Vectorize and unroll
    reg_load.tag_vector_level(ffout, FOUT_BLOCKING);
    conv.tag_vector_level(ffout, FOUT_BLOCKING);
#if PATTERN_0
    conv_P0.tag_vector_level(ffout, FOUT_BLOCKING);
#endif
    reg_store.tag_vector_level(ffout, FOUT_BLOCKING);

    conv.tag_unroll_level(xx);
    conv.tag_unroll_level(fout_b);
#if PATTERN_0
    conv_P0.tag_unroll_level(xx);
    conv_P0.tag_unroll_level(fout_b);
#endif

    reg_load.tag_unroll_level(xx);
    reg_load.tag_unroll_level(fout_b);

    reg_store.tag_unroll_level(xx);
    reg_store.tag_unroll_level(fout_b);

    // schedule for conv_conclude
    // This schedule is the same as conv computation
    computation reg_load_conclude(
        "reg_load_conclude",
        {n, y, fin_b, fout_b, ffout, x_conclude},
        conv_init(n, y, fout_b, x_conclude, ffout)
    );

    computation reg_store_conclude(
        "reg_store_conclude",
        {n, y, fin_b, fout_b, ffout, x_conclude},
        conv_conclude(n, y, fin_b, 0, 0, 0, fout_b, ffout, x_conclude)
    );

    reg_load_conclude.tag_vector_level(ffout, FOUT_BLOCKING);
    conv_conclude.tag_vector_level(ffout, FOUT_BLOCKING);
    reg_store_conclude.tag_vector_level(ffout, FOUT_BLOCKING);

    conv_conclude.tag_unroll_level(x_conclude);
    conv_conclude.tag_unroll_level(fout_b);

    reg_load_conclude.tag_unroll_level(x_conclude);
    reg_load_conclude.tag_unroll_level(fout_b);

    reg_store_conclude.tag_unroll_level(x_conclude);
    reg_store_conclude.tag_unroll_level(fout_b);

    // Parallelize and order
    conv.tag_parallel_level(y);
    conv.tag_parallel_level(n);

    conv_init.then(reg_load, y)
#if PATTERN_0
             .then(conv_P0, x_b)
#endif
             .then(conv, x_b)
             .then(reg_store, x_b)
             .then(reg_load_conclude, y)
             .then(conv_conclude, fin_b)
             .then(reg_store_conclude, fin_b);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv_buf("conv_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);
    
    // This is where intermediate results of convolution will be stored.
    // We rely on the compiler to detect that this buffer can be mapped to CPU registers.
    buffer reg_buf("reg_buf", {FOUT_NB_BLOCKS, X_BLOCKING, FOUT_BLOCKING}, p_float32, a_temporary);

    conv_init.store_in(&conv_buf, {n, fout_b, y, x, ffout});
    conv_out.store_in(&reg_buf, {fout_b, x%X_BLOCKING, ffout});

    reg_load.store_in(&reg_buf, {fout_b, x_bound%X_BLOCKING, ffout});
    conv.store_in(&reg_buf, {fout_b, x_bound%X_BLOCKING, ffout});
#if PATTERN_0
    conv_P0.store_in(&reg_buf, {fout_b, x_bound%X_BLOCKING, ffout});
#endif
    reg_store.store_in(&conv_buf, {n, fout_b, y, x_bound, ffout});

    reg_load_conclude.store_in(&reg_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    conv_conclude.store_in(&reg_buf, {fout_b, x_conclude%X_BLOCKING, ffout});
    reg_store_conclude.store_in(&conv_buf, {n, fout_b, y, x_conclude, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(), 
        filter.get_buffer(), 
        filter2.get_buffer(),
        bias.get_buffer(), 
        &conv_buf
    },"generated_conv_layer.o");

    return 0;
}
