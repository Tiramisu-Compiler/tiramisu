#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main()
{
    init("densenet_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var x("x", 0, N), y("y", 0, N), z("z", 0, 4*GR), n("n", 0, BATCH_SIZE);
    var x_pad("x", 0, N + 2), y_pad("y", 0, N + 2);
    var k_x("k_x", 0, K_X), k_y("k_y", 0, K_Y), fout("fout", 0, GR);

    var z_b("z_b", 0, Z_NB_BLOCKS), zz("zz", 0, Z_BLOCKING);
    var fout_b("fout_b", 0, FOUT_NB_BLOCKS), ffout("ffout", 0, FOUT_BLOCKING);

    input c_input("c_input", {n, z_b, y_pad, x_pad, zz}, p_float32);

    // Convolution computation
    input conv_filter("conv_filter", {z_b, fout_b, k_y, k_x, ffout, zz}, p_float32);
    input conv_bias("conv_bias", {fout}, p_float32);

    computation init_output("init_output", {n, fout_b, y, x, ffout}, conv_bias(fout_b*FOUT_BLOCKING + ffout));
    computation conv(
        "conv", 
        {n, z_b, fout_b, y, x, k_y, k_x, ffout, zz}, 
        init_output(n, fout_b, y, x, ffout) + c_input(n, z_b, y + k_y, x + k_x, zz)*conv_filter(z_b, fout_b, k_y, k_x, ffout, zz)
    );
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    init_output.then(conv, n);

    //n, z_b, fout_b, y, x, k_y, k_x, ffout, zz
    conv.interchange(x, k_y);
    conv.interchange(x, k_x);
    conv.interchange(x, ffout);

    conv.interchange(y, k_y);
    conv.interchange(y, k_x);
    conv.interchange(y, ffout);
    //n, z_b, fout_b, k_y, k_x, ffout, y, x, zz
    
    conv.tag_parallel_level(n);
    conv.vectorize(ffout, FOUT_BLOCKING);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer output_buf("output_buf", {BATCH_SIZE, GR/FOUT_BLOCKING, N, N, FOUT_BLOCKING}, p_float32, a_output);

    init_output.store_in(&output_buf);
    conv.store_in(&output_buf, {n, fout_b, y, x, ffout});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_input.get_buffer(),
        conv_filter.get_buffer(), 
        conv_bias.get_buffer(), 
        &output_buf
    }, "densenet_block_tiramisu.o");

    return 0;
}
