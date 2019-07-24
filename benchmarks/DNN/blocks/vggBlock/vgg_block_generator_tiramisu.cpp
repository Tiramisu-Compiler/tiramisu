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

    var fin1_b("fin1_b", 0, FIN1_NB_BLOCKS), ffin1("ffin1", 0, FIN1_BLOCKING);
    var fin2_b("fin2_b", 0, FIN2_NB_BLOCKS), ffin2("ffin2", 0, FIN2_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    // Input computations
    input c_input("c_input", {n, fin1_b, y, x, ffin1}, p_float32);

    input bias1("bias1", {fin2_b, ffin2}, p_float32);
    input filter1("filter1", {fin2_b, fin1_b, k_y, k_x, ffin1, ffin2}, p_float32);

    input bias2("bias2", {fout_b, ffout}, p_float32);
    input filter2("filter2", {fout_b, fin2_b, k_y, k_x, ffin2, ffout}, p_float32);

    // Pad input
    computation init_input_padded("init_input_padded", {n, fin1_b, y_pad, x_pad, ffin1}, cast(p_float32, 0));
    computation copy_input("copy_input", {n, fin1_b, y, x, ffin1}, c_input(n, fin1_b, y, x, ffin1));
    view input_padded("input_padded", {n, fin1_b, y_pad, x_pad, ffin1}, p_float32);

    // First conv computations
    computation init_output1("init_output1", {n, fin2_b, y_pad, x_pad, ffin2}, cast(p_float32, 0));

    computation conv1_init("conv1_init", {n, fin2_b, y, x, ffin2}, bias1(fin2_b, ffin2));
    computation conv1(
        "conv1", 
        {n, fin2_b, y, x, fin1_b, k_y, k_x, ffin1, ffin2}, 
        conv1_init(n, fin2_b, y, x, ffin2) + filter1(fin2_b, fin1_b, k_y, k_x, ffin1, ffin2) * input_padded(n, fin1_b, y + k_y, x + k_x, ffin1)
    );

    // First relu
    computation relu1(
        "relu1", 
        {n, fin2_b, y, x, ffin2}, 
        expr(
            o_max,
            0.f,
            conv1(n, fin2_b, y, x, 0, 0, 0, 0, ffin2)
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
    init_input_padded.then(copy_input, fin1_b)
                     .then(maxpool_init, n)
                     .then(init_output1, n)
                     .then(conv1_init, fin2_b)
                     .then(conv1, y)
                     .then(relu1, y)
                     .then(conv2_init, n)
                     .then(conv2, y)
                     .then(maxpool, y);

    copy_input.vectorize(ffin1, FIN1_BLOCKING);
    
    conv1.interchange(x, fin1_b);
    conv1.interchange(x, k_y);
    conv1.interchange(x, k_x);

    conv1.vectorize(ffin2, FIN2_BLOCKING);
    relu1.vectorize(ffin2, FIN2_BLOCKING);    
    
    conv2.interchange(x, fin2_b);
    conv2.interchange(x, k_y);
    conv2.interchange(x, k_x);

    conv2.vectorize(ffout, FOUT_BLOCKING);
    maxpool.vectorize(ffout, FOUT_BLOCKING);

    maxpool.tag_parallel_level(n);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer padded_input_buf("padded_input_buf", {BATCH_SIZE, FIN1_NB_BLOCKS, N + 2, N + 2, FIN1_BLOCKING}, p_float32, a_temporary);
    buffer conv1_buf("conv1_buf", {BATCH_SIZE, FIN2_NB_BLOCKS, N + 2, N + 2, FIN2_BLOCKING}, p_float32, a_temporary);
    buffer conv2_buf("conv2_buf", {BATCH_SIZE, N, FOUT_BLOCKING}, p_float32, a_temporary);
    buffer maxpool_buf("maxpool_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N/2, N/2, FOUT_BLOCKING}, p_float32, a_output);

    init_input_padded.store_in(&padded_input_buf);
    copy_input.store_in(&padded_input_buf, {n, fin1_b, y + 1, x + 1, ffin1});
    input_padded.store_in(&padded_input_buf);

    init_output1.store_in(&conv1_buf);

    conv1_init.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});
    conv1.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});
    relu1.store_in(&conv1_buf, {n, fin2_b, y + 1, x + 1, ffin2});

    relu1_padded.store_in(&conv1_buf);

    conv2_init.store_in(&conv2_buf, {n, x, ffout});
    conv2.store_in(&conv2_buf, {n, x, ffout});

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
        &maxpool_buf
    }, "generated_vgg_block.o");

    return 0;
}
