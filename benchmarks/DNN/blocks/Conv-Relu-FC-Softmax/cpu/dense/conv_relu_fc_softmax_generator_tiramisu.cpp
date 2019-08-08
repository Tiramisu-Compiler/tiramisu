#include <tiramisu/tiramisu.h>
#include "configure.h"
using namespace tiramisu;

#define N_X N
#define N_Y N

#define K_X K
#define K_Y K

#define OUT_WIDTH_CST ((N_X - K_X + 2 * PADDING)/STRIDE +1)
#define OUT_HEIGHT_CST ((N_Y - K_Y + 2 * PADDING)/STRIDE +1)
int main(int argc, char **argv)
{
	tiramisu::init("conv_relu_fc_softmax");

	// -------------------------------------------------------
	// Layer I
	// -------------------------------------------------------

	//Constants
	constant padded_width("padded_width", expr(N_X) + 2*expr(PADDING));
	constant padded_height("padded_height", expr(N_Y) + 2*expr(PADDING));

	constant output_width("output_width", cast(p_int32, expr(OUT_WIDTH_CST)));
	constant output_height("output_height", cast(p_int32, expr(OUT_HEIGHT_CST)));

	constant fc_input_size("fc_input_size", expr(FOut) * output_height * output_width);
	constant fc_output_size("fc_output_size", expr(FC_OUTPUT_SIZE));

	//Iteration variables
	var fin("fin", 0, FIn);
	var x_padded("x_padded", 0, padded_width), y_padded("y_padded", 0, padded_height);
	var kx("kx", 0, K_X), ky("ky", 0, K_Y), fout("fout", 0, FOut), b("b", 0, BATCH_SIZE);
	var x_output("x_output", 0, output_width), y_output("y_output", 0, output_height);

	var fc_input("fc_input", 0, fc_input_size);
	var fc_output("fc_output", 0, fc_output_size);

	//Inputs
	input input_batch("input_batch", {b, fin, y_padded, x_padded}, p_float32);
	input input_filter("input_filter", {fout, fin, ky, kx}, p_float32);
	input input_bias("input_bias", {fout}, p_float32);

	input input_FC_weights("input_FC_weights", {fc_output, fc_input}, p_float32);
	input input_FC_bias("input_FC_bias", {fc_output}, p_float32);

	//Computations
	//Flattened Convolution
	computation result_init("result_init", {b, fout, y_output, x_output}, cast(p_float32, input_bias(fout)));
	computation convolve("convolve", {b, fout, y_output, x_output, fin, ky, kx}, p_float32);
	convolve.set_expression(expr(result_init(b, fout, y_output, x_output) + input_filter(fout, fin, ky, kx) * input_batch(b, fin, y_output * STRIDE + ky, x_output * STRIDE + kx)));

	//Relu
	computation relu("relu", {b, fc_input}, p_float32);
	relu.set_expression(expr(o_max, cast(p_float32, 0), relu(b, fc_input)));

	//FC
	computation result_FC_init("result_FC_init", {b, fc_output}, cast(p_float32, input_FC_bias(fc_output)));
	computation FC("FC", {b, fc_input, fc_output}, p_float32);
	FC.set_expression(expr(FC(b, fc_input-1, fc_output) + input_FC_weights(fc_output, fc_input) * relu(b, fc_input)));

	//Softmax
	computation expo("expo", {b, fc_output}, expr(o_expo, FC(b, fc_input_size-1, fc_output)));
	computation init_sum_expo("init_sum_expo", {b, fc_output}, cast(p_float32, 0));
	computation sum_expo("sum_expo", {b, fc_output}, p_float32);
	sum_expo.set_expression(expr(sum_expo(b, fc_output-1) + expo(b, fc_output)));
	computation softmax("softmax", {b, fc_output}, expr(expo(b, fc_output) / sum_expo(b, fc_output_size-1)));

	// -------------------------------------------------------
	// Layer IIl
	// -------------------------------------------------------
	result_init.then(convolve, computation::root)
		.then(relu, computation::root)
		.then(result_FC_init, computation::root)
		.then(FC, computation::root)
		.then(expo, computation::root)
		.then(init_sum_expo, computation::root)
		.then(sum_expo, computation::root)
		.then(softmax, computation::root);

	// -------------------------------------------------------
	// Layer III
	// -------------------------------------------------------

	//Input Buffers
	buffer buf_input_batch("buf_input_batch", {BATCH_SIZE, FIn, padded_height, padded_width}, p_float32, a_input);
	buffer buf_input_filter("buf_input_filter", {FOut, FIn, K_Y, K_X}, p_float32, a_input);
	buffer buf_bias("buf_bias", {FOut}, p_float32, a_input);

	buffer buf_fc_weights("buf_fc_weights", {fc_output_size, fc_input_size}, p_float32, a_input);
	buffer buf_fc_bias("buf_fc_bias", {fc_output_size}, p_float32, a_input);

	//Temporary Buffers
	buffer buf_conv_result_flattened("buf_conv_result_flattened", {BATCH_SIZE, fc_input_size}, p_float32, a_temporary);

	buffer buf_sum_expo("buf_sum_expo", {BATCH_SIZE}, p_float32, a_temporary);

	//Output Buffers
	buffer buf_result("buf_result", {BATCH_SIZE, fc_output_size}, p_float32, a_output);

	//Store inputs
	input_batch.store_in(&buf_input_batch, {b, fin, y_padded, x_padded});
	input_filter.store_in(&buf_input_filter, {fout, fin, ky, kx});
	input_bias.store_in(&buf_bias, {fout});

	input_FC_weights.store_in(&buf_fc_weights, {fc_output, fc_input});
	input_FC_bias.store_in(&buf_fc_bias, {fc_output});

	//Store computations
	result_init.store_in(&buf_conv_result_flattened, {b, x_output + y_output * OUT_WIDTH_CST + fout * OUT_WIDTH_CST * OUT_HEIGHT_CST});
	convolve.store_in(&buf_conv_result_flattened, {b, x_output + y_output * OUT_WIDTH_CST + fout * OUT_WIDTH_CST * OUT_HEIGHT_CST});

	relu.store_in(&buf_conv_result_flattened, {b, fc_input});

	result_FC_init.store_in(&buf_result, {b, fc_output});
	FC.store_in(&buf_result, {b, fc_output});

	expo.store_in(&buf_result, {b, fc_output});
	init_sum_expo.store_in(&buf_sum_expo, {b});
	sum_expo.store_in(&buf_sum_expo, {b});

	softmax.store_in(&buf_result, {b, fc_output});

	// -------------------------------------------------------
	// Code Generation
	// -------------------------------------------------------
	codegen({&buf_input_batch, &buf_input_filter, &buf_bias, &buf_fc_weights, &buf_fc_bias, &buf_result}, "conv_relu_fc_softmax_generator_tiramisu.o");

	return 0;
}
