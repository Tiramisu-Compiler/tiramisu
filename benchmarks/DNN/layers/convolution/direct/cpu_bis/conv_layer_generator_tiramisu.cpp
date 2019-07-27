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
    conv.vectorize(ffout, FOUT_BLOCKING);
    
    conv.tag_unroll_level(fin);
    conv.tag_parallel_level(n);
    
    conv_init.then(conv, x);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    buffer conv_buf("conv_buf", {BATCH_SIZE, FOUT_NB_BLOCKS, N, N, FOUT_BLOCKING}, p_float32, a_output);

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
