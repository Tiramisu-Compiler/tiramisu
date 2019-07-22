#include <tiramisu/tiramisu.h>
#include "configure.h"
using namespace tiramisu;

#define N_X N
#define N_Y N

#define K_X K
#define K_Y K

int main(int argc, char **argv)
{
	init("transpose_conv");
	
	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------
	
	//Constants
	constant input_width("input_width", expr(N_X));
	constant input_height("input_height", expr(N_Y));
	
	constant output_width("output_width", expr(STRIDE)*(expr(N_X) - expr(1)) + expr(K_X));
	constant output_height("output_height", expr(STRIDE)*(expr(N_Y) - expr(1)) + expr(K_Y));
	
	//Iteration variables
	var fin("fin", 0, FIn);
	var x("x", 0, input_width), y("y", 0, input_height);
	var kx("kx", 0, K_X), ky("ky", 0, K_Y), fout("fout", 0, FOut), b("b", 0, BATCH_SIZE);
	var x_output("x_output", 0, output_width), y_output("y_output", 0, output_height);
	
	//Inputs
	input input_batch("input_batch", {b, fin, y, x}, p_float32);
	input input_filter("input_filter", {fout, fin, ky, kx}, p_float32);
	input input_bias("input_bias", {fout}, p_float32);
	
	//Computations
	computation result_init("result_init", {b, fout, y_output, x_output}, expr(cast(p_float32, input_bias(fout))));
	
	computation upsample("upsample", {b, fout, y, x, fin, ky, kx}, p_float32);
	upsample.set_expression(expr(upsample(b, fout, y, x, fin, ky, kx) + input_filter(fout, fin, ky, kx) * input_batch(b, fin, y, x)));
	
	// -------------------------------------------------------
	// Layer II
	// -------------------------------------------------------
	result_init.then(upsample, fout);
	upsample.parallelize(b);
	
	if (N_X>=32)
		upsample.vectorize(x, 32);
	
	if (STRIDE * (N_X - 1) - K + 2 * PADDING >=32)
		result_init.vectorize(x_output, 32);
	
	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------
	
	//Input Buffers
	buffer buf_input_batch("buf_input_batch", {BATCH_SIZE, FIn, input_height, input_width}, p_float32, a_input);
	buffer buf_input_filter("buf_input_filter", {FOut, FIn, K_Y, K_X}, p_float32, a_input);
	buffer buf_bias("buf_bias", {FOut}, p_float32, a_input);
	
	//Output Buffers
	buffer buf_result("buf_result", {BATCH_SIZE, FOut, output_height, output_width}, p_float32, a_output);
	
	//Store inputs
	input_batch.store_in(&buf_input_batch, {b, fin, y, x});
	input_filter.store_in(&buf_input_filter, {fout, fin, ky, kx});
	input_bias.store_in(&buf_bias, {fout});
	
	//Store computations
	result_init.store_in(&buf_result, {b, fout, y_output, x_output});
	upsample.store_in(&buf_result, {b, fout, STRIDE * y + ky, STRIDE * x + kx});
	
	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	codegen({&buf_input_batch, &buf_input_filter, &buf_bias, &buf_result}, "transpose_conv_generator_tiramisu.o");
	
	return 0;
}
