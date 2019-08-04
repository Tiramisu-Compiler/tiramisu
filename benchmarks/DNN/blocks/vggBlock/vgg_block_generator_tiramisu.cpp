#include <tiramisu/tiramisu.h>
#include "configure.h"
#define SCHEDULE_CPU 1

using namespace tiramisu;

int main(int argc, char **argv)
{
    init("vgg_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), n("n", 0, BATCH_SIZE);
    var x_pad("x_pad", 0, N + 2), y_pad("y_pad", 0, N + 2);

    var o_x("o_x", 0, N/2), o_y("o_y", 0, N/2);
    var k_x("k_x", 0, K), k_y("k_y", 0, K);

    var fin1("fin1", 0, FIn);
    var fin2_b("fin2_b", 0, FIN2_NB_BLOCKS), ffin2("ffin2", 0, FIN2_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    // Input computations
    input c_input("c_input", {n, y_pad, x_pad, fin1}, p_float32);

    input bias1("bias1", {fin2_b, ffin2}, p_float32);
    input filter1("filter1", {fin2_b, k_y, k_x, fin1, ffin2}, p_float32);

    input bias2("bias2", {fout_b, ffout}, p_float32);
    input filter2("filter2", {fout_b, fin2_b, k_y, k_x, ffin2, ffout}, p_float32);

    // First conv computations
    computation zero_conv1("zero_conv2", {n, fin2_b, y_pad, x_pad, ffin2}, cast(p_float32, 0));

    computation conv1_init("conv1_init", {n, fin2_b, y, x, ffin2}, bias1(fin2_b, ffin2));
    computation conv1(
        "conv1", 
        {n, fin2_b, y, x, k_y, k_x, fin1, ffin2}, 
        conv1_init(n, fin2_b, y, x, ffin2) + filter1(fin2_b, k_y, k_x, fin1, ffin2) * c_input(n, y + k_y, x + k_x, fin1)
    );

    // First relu
    computation relu1(
        "relu1", 
        {n, fin2_b, y, x, ffin2}, 
        expr(
            o_max,
            0.f,
            conv1(n, fin2_b, y, x, 0, 0, 0, ffin2)
        )
    );

    view relu1_padded("relu1_padded", {n, fin2_b, y_pad, x_pad, ffin2}, p_float32);

    // Second conv computations
    computation conv2_init("conv2_init", {n, fout_b, y, x, ffout}, bias2(fout_b, ffout));
    computation conv2(
        "conv2", 
        {n, fout_b, y, x, fin2_b, k_y, k_x, ffin2, ffout}, 
        conv2_init(n, fout_b, y, x, ffout) + filter2(fout_b, fin2_b, k_y, k_x, ffin2, ffout) * relu1_padded(n, fin2_b, y + k_y, x + k_x, ffin2)
    );

    // Maxpooling computation
    computation maxpool_init("maxpool_init", {n, fout_b, o_y, o_x, ffout}, 0.f);

    view c_output("c_output", {n, fout_b, y, x, ffout}, p_float32);
    computation maxpool(
        "maxpool",
        {n, fout_b, y, x, ffout}, 
        expr(
            o_max, 
            c_output(n, fout_b, y, x, ffout), 
            conv2(n, fout_b, y, x, 0, 0, 0, 0, ffout)
        )
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------    
    // Schedule for first convolution
    var y1_b("y1_b", 0, Y1_NB_BLOCKS), x1_b("x1_b", 0, X1_NB_BLOCKS);
    var yy1, xx1;
    
    // Loop through weights to load them into cache
    computation prefetch_weights1(
        "prefetch_weights1",
        {n, fout_b, y1_b, x1_b, k_y, k_x, fin1, ffout},
        filter1(fout_b, k_y, k_x, fin1, ffout),
        SCHEDULE_PREFETCH_WEIGHTS1
    );

    if (N >= 224) {
        conv1_init.tile(y, x, Y1_BLOCKING, X1_BLOCKING, y1_b, x1_b, yy1, xx1);
        conv1.tile(y, x, Y1_BLOCKING, X1_BLOCKING, y1_b, x1_b, yy1, xx1);
        relu1.tile(y, x, Y1_BLOCKING, X1_BLOCKING, y1_b, x1_b, yy1, xx1);
        
        // n, fin2_b, y1_b, x1_b, yy1, xx1, k_y, k_x, fin, ffin2
        conv1.interchange(xx1, k_y);
        conv1.interchange(xx1, k_x);
        // n, fin2_b, y1_b, x1_b, yy1, k_y, k_x, xx1, fin, ffin2
        conv1.interchange(yy1, k_y);
        conv1.interchange(yy1, k_x);
        // n, fin2_b, y1_b, x1_b, k_y, k_x, yy1, xx1, fin, ffin2
    }
    
    else {
        // n, fin2_b, y, x, k_y, k_x, fin, ffin2
        conv1.interchange(x, k_y);
        
        conv1.split(x, X1_BLOCKING, x1_b, xx1);
        conv1.interchange(xx1, k_x);
        // n, fin2_b, y, k_y, x1_b, k_x, xx1, fin, ffin2
    }

    conv1_init.vectorize(ffin2, VEC_LEN);
    conv1.vectorize(ffin2, VEC_LEN);
    relu1.vectorize(ffin2, VEC_LEN);
    
    // Schedule for second convolution
    var y2_b("y2_b", 0, Y2_NB_BLOCKS), x2_b("x2_b", 0, X2_NB_BLOCKS);
    var yy2, xx2;

    // Loop through weights to load them into cache
    computation prefetch_weights2(
        "prefetch_weights2",
        {n, fout_b, y2_b, x2_b, fin2_b, k_y, k_x, ffin2, ffout},
        filter2(fout_b, fin2_b, k_y, k_x, ffin2, ffout)
    );

    conv2_init.tile(y, x, Y2_BLOCKING, X2_BLOCKING, y2_b, x2_b, yy2, xx2);
    conv2.tile(y, x, Y2_BLOCKING, X2_BLOCKING, y2_b, x2_b, yy2, xx2);
    maxpool.tile(y, x, Y2_BLOCKING, X2_BLOCKING, y2_b, x2_b, yy2, xx2);
        
    // n, fout_b, y2_b, x2_b, yy2, xx2, fin2_b, k_y, k_x, ffin, ffout
    conv2.interchange(xx2, fin2_b);
    conv2.interchange(xx2, k_y);
    conv2.interchange(xx2, k_x);
    // n, fout_b, y2_b, x2_b, yy2, fin2_b, k_y, k_x, xx2, ffin, ffout
    conv2.interchange(yy2, fin2_b);
    conv2.interchange(yy2, k_y);
    conv2.interchange(yy2, k_x);
    // n, fout_b, y2_b, x2_b, fin2_b, k_y, k_x, yy2, xx2, ffin, ffout
    
    conv2_init.vectorize(ffout, VEC_LEN);
    conv2.vectorize(ffout, VEC_LEN);
    maxpool.vectorize(ffout, VEC_LEN);

    maxpool.tag_parallel_level(n);

    if (SCHEDULE_PREFETCH_WEIGHTS1) {
        maxpool_init.then(zero_conv1, n)
                    .then(conv1_init, fin2_b)
                    .then(prefetch_weights1, x1_b)
                    .then(conv1, x1_b)
                    .then(relu1, x1_b)
                    .then(conv2_init, n)
                    .then(prefetch_weights2, x2_b)
                    .then(conv2, fin2_b)
                    .then(maxpool, x2_b);
    }

    else {
        maxpool_init.then(zero_conv1, n)
                    .then(conv1_init, fin2_b)
                    .then(conv1, y)
                    .then(relu1, y)
                    .then(conv2_init, n)
                    .then(prefetch_weights2, x2_b)
                    .then(conv2, fin2_b)
                    .then(maxpool, x2_b);
    }

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv1_buf("conv1_buf", {BATCH_SIZE, FIN2_NB_BLOCKS, N + 2, N + 2, FIN2_BLOCKING}, p_float32, a_input);
    buffer conv2_buf("conv2_buf", {BATCH_SIZE, N, X2_BLOCKING, FOUT_BLOCKING}, p_float32, a_input);
    buffer maxpool_buf("maxpool_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    buffer prefetch_w_buf("prefetch_w_buf", {1}, p_float32, a_temporary);
    if (SCHEDULE_PREFETCH_WEIGHTS1)
        prefetch_weights1.store_in(&prefetch_w_buf, {});

    prefetch_weights2.store_in(&prefetch_w_buf, {});

    zero_conv1.store_in(&conv1_buf);
    conv1_init.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});
    conv1.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});
    relu1.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});

    relu1_padded.store_in(&conv1_buf);

    conv2_init.store_in(&conv2_buf, {n, y, x%X2_BLOCKING, ffout});
    conv2.store_in(&conv2_buf, {n, y, x%X2_BLOCKING, ffout});

    maxpool_init.store_in(&maxpool_buf);

    c_output.store_in(&maxpool_buf, {n, fout_b, y/2, x/2, ffout});
    maxpool.store_in(&maxpool_buf, {n, fout_b, y/2, x/2, ffout});

    // -------------------------------------------------------
    // Code generation
    // -------------------------------------------------------
    tiramisu::codegen({
        c_input.get_buffer(), 
        filter1.get_buffer(), 
        bias1.get_buffer(), 
        filter2.get_buffer(), 
        bias2.get_buffer(),
        &conv1_buf,
        &conv2_buf,
        &maxpool_buf
    }, "generated_vgg_block.o");

    return 0;
}
