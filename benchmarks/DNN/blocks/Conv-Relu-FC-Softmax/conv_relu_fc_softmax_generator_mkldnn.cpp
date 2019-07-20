#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "mkldnn.hpp"
#include "mkldnn_debug.h"

#include "configure.h"

using namespace mkldnn;

void conv_relu_fc_softmax()
{
	// Conv output size
	int OUTPUT_N = (N - K + 2 * PADDING)/STRIDE + 1;
	// Conv output flattened size (FC input size)
	int FC_INPUT_SIZE = OUTPUT_N * OUTPUT_N * FOut;
	std::vector<std::chrono::duration<double, std::milli>> duration_vector;

	engine cpu_engine(engine::kind::cpu, 0);
	stream cpu_stream(cpu_engine);

	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;

	//Initialize buffers
	memory::dims strides = {STRIDE, STRIDE};
	memory::dims paddings = {PADDING, PADDING};

	std::vector<float> buf_input_batch(BATCH_SIZE*FIn*N*N);
	std::vector<float> buf_filter(FOut*FIn*K*K);
	std::vector<float> buf_bias(FOut);

	std::vector<float> buf_fc_weights(FC_OUTPUT_SIZE * FC_INPUT_SIZE);
	std::vector<float> buf_fc_bias(FC_OUTPUT_SIZE);

	std::vector<float> buf_result(BATCH_SIZE * FC_OUTPUT_SIZE);
	for (int oc = 0; oc < FOut; oc++)
		for (int ic = 0; ic < FIn; ic++)
			for (int ky = 0; ky < K; ky++)
				for (int kx = 0; kx < K; kx++)
					buf_filter[kx + ky*K + ic*K*K + oc*K*K*FIn] = 0.001f;

	for (int oc = 0; oc < FOut; oc++)
		buf_bias[oc] = 0.001f;

	for (int b = 0; b < BATCH_SIZE; b++)
		for (int ic = 0; ic < FIn; ic++)
			for (int y = 0; y < N; y++)
				for (int x = 0; x < N; x++)
					buf_input_batch[x + y*N + ic*N*N + b*N*N*FIn] = 0.001f;

	float val=0.000001;
	for (int y=0; y < FC_OUTPUT_SIZE; y++)
		for (int x=0; x < FC_INPUT_SIZE; x++){
			buf_fc_weights[x + y * FC_INPUT_SIZE] = val;
			val+=0.0000001;
		}

	for (int z = 0; z < FC_OUTPUT_SIZE; z++)
		buf_fc_bias[z] = 0.001;

	// Create memory objects with user data format
	auto conv_weights_user_md = memory::desc(
		{FOut, FIn, K, K},
		memory::data_type::f32,
		memory::format_tag::oihw
	);

	auto conv_bias_user_md = memory::desc(
		{FOut},
		memory::data_type::f32,
		memory::format_tag::x
	);

	auto conv_weights_user_memory = memory(conv_weights_user_md, cpu_engine, buf_filter.data());
	auto conv_bias_user_memory = memory(conv_bias_user_md, cpu_engine, buf_bias.data());

	// Create memory objects with a data format selected by the convolution primitive
	auto conv_src_md = memory::desc(
		{BATCH_SIZE, FIn, N, N},
		memory::data_type::f32,
		memory::format_tag::any
	);

	auto conv_weights_md = memory::desc(
		{FOut, FIn, K, K},
		memory::data_type::f32,
		memory::format_tag::any
	);

	auto conv_bias_md = memory::desc(
		{FOut},
		memory::data_type::f32,
		memory::format_tag::any
	);

	auto output_md = memory::desc(
		{BATCH_SIZE, FOut, OUTPUT_N, OUTPUT_N},
		memory::data_type::f32,
		memory::format_tag::any
	);

	// Create the convolution primitive descriptor
	auto conv_desc = convolution_forward::desc(
		prop_kind::forward_inference,
		algorithm::convolution_direct,
		conv_src_md,
		conv_weights_md,
		conv_bias_md,
		output_md,
		strides,
		paddings,
		paddings
	);

	auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, cpu_engine);

	auto input_user_md = memory::desc(
		{BATCH_SIZE, FIn, N, N},
		memory::data_type::f32,
		memory::format_tag::nchw
	);

	auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), cpu_engine);
	auto input_user_memory = memory(input_user_md, cpu_engine, buf_input_batch.data());

	auto input_memory = memory(conv_prim_desc.src_desc(), cpu_engine);

	// Convert the user's input to the primitive's format
	net.push_back(reorder(input_user_memory, input_memory));
	net_args.push_back({
		{MKLDNN_ARG_FROM, input_user_memory},
		{MKLDNN_ARG_TO, input_memory}
	});

	// Convert user weights format to primitive's format if they're different
	auto conv_weights_memory = conv_weights_user_memory;
	if (conv_prim_desc.weights_desc() != conv_weights_user_memory.get_desc()) {
		conv_weights_memory = memory(conv_prim_desc.weights_desc(), cpu_engine);
		reorder(conv_weights_user_memory, conv_weights_memory)
			.execute(cpu_stream, conv_weights_user_memory, conv_weights_memory);
	}

	// Create convolution primitive
	net.push_back(convolution_forward(conv_prim_desc));
	net_args.push_back({
		{MKLDNN_ARG_SRC, input_memory},
		{MKLDNN_ARG_WEIGHTS, conv_weights_memory},
		{MKLDNN_ARG_BIAS, conv_bias_user_memory},
		{MKLDNN_ARG_DST, conv_dst_memory}
	});

	// Applying RELU to conv layer's output
	auto relu_desc = eltwise_forward::desc(
		prop_kind::forward_inference,
		algorithm::eltwise_relu,
		conv_dst_memory.get_desc(),
		0
	);

	auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, cpu_engine);
	net.push_back(eltwise_forward(relu_prim_desc));
	net_args.push_back({
		{MKLDNN_ARG_SRC, conv_dst_memory},
		{MKLDNN_ARG_DST, conv_dst_memory}
	});

	// FC layer
	memory::dims fc_src_tz = {BATCH_SIZE, FC_INPUT_SIZE};
	memory::dims fc_weights_tz = {FC_OUTPUT_SIZE, FC_INPUT_SIZE};
	memory::dims fc_bias_tz = {FC_OUTPUT_SIZE};
	memory::dims fc_dst_tz = {BATCH_SIZE, FC_OUTPUT_SIZE};

	// Create memory for user data
	auto fc_user_weights_memory = memory({
		{fc_weights_tz}, memory::data_type::f32, memory::format_tag::nc}, cpu_engine, buf_fc_weights.data());

	auto fc_user_bias_memory = memory(
		{{fc_bias_tz}, memory::data_type::f32, memory::format_tag::x}, cpu_engine, buf_fc_bias.data());

	auto user_dst_memory = memory(
		{{fc_dst_tz}, memory::data_type::f32, memory::format_tag::nc}, cpu_engine, buf_result.data());

	// Create memory descriptors for FC layer data selected by the primitive
	// FC input layer format for flattening
	auto fc_src_md = memory::desc({
		{fc_src_tz},
		memory::data_type::f32,
		memory::format_tag::any
	});

	auto fc_bias_md = memory::desc({
		{fc_bias_tz},
		memory::data_type::f32,
		memory::format_tag::any
	});

	auto fc_weights_md = memory::desc({
		{fc_weights_tz },
		memory::data_type::f32,
		memory::format_tag::any
	});
	auto fc_dst_md = memory::desc({
		{fc_dst_tz},
		memory::data_type::f32,
		memory::format_tag::any
	});
	// create an FC layer
	auto fc_desc = inner_product_forward::desc(
		prop_kind::forward_inference,
		fc_src_md,
		fc_weights_md,
		fc_bias_md,
		fc_dst_md
	);

	// Convert FC input weights to primitive's format if they're different
	auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, cpu_engine);
	auto fc_weights_memory = fc_user_weights_memory;
	if (fc_prim_desc.weights_desc() != fc_user_weights_memory.get_desc()) {
		fc_weights_memory = memory(fc_prim_desc.weights_desc(), cpu_engine);
		reorder(fc_user_weights_memory, fc_weights_memory)
			.execute(cpu_stream, fc_user_weights_memory, fc_weights_memory);
	}
	auto fc_dst_memory = memory(fc_prim_desc.dst_desc(), cpu_engine);

	// Create FC primitive and add it to net
	net.push_back(inner_product_forward(fc_prim_desc));
	net_args.push_back({
		{MKLDNN_ARG_SRC, conv_dst_memory},
		{MKLDNN_ARG_WEIGHTS, fc_weights_memory},
		{MKLDNN_ARG_BIAS, fc_user_bias_memory},
		{MKLDNN_ARG_DST, fc_dst_memory}
	});

	// create reorder between internal and user data if it is needed and add it to the network
	if (fc_dst_memory != user_dst_memory) {
		net.push_back(reorder(fc_dst_memory, user_dst_memory));
		net_args.push_back({
			{MKLDNN_ARG_FROM, fc_dst_memory},
			{MKLDNN_ARG_TO, user_dst_memory}
		});
	}
	// Adding softmax
	auto softmax_desc = softmax_forward::desc(
		prop_kind::forward_inference,
		user_dst_memory.get_desc(),
		1
	);
	auto softmax_prim_desc = softmax_forward::primitive_desc(softmax_desc, cpu_engine);
	// create convolution primitive and add it to net
	net.push_back(softmax_forward(softmax_prim_desc));
	net_args.push_back({
		{MKLDNN_ARG_SRC, user_dst_memory},
		{MKLDNN_ARG_DST, user_dst_memory}
	});

	// Execute the network
	for (int i = 0; i < NB_TESTS; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int j=0; j < net.size(); j++)
			net[j].execute(cpu_stream, net_args[j]);

		cpu_stream.wait();

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end - start;
		duration_vector.push_back(duration);
	}

	std::cout << "\n\n\tMKLDNN Conv-Relu-FC-Softmax time : " << median(duration_vector) << " ms." << std::endl;

	if (SHOW_OUTPUT)
	{
		for(int n=0; n<BATCH_SIZE; n++){
			for(int z=0; z<FC_OUTPUT_SIZE; z++)
				std::cout << buf_result[z + n * FC_OUTPUT_SIZE] << ", ";
			std::cout << std::endl;
		}
	}

	if (WRITE_RESULT_TO_FILE){
		// Write results to file
		FILE* f = fopen("mkldnn_result.txt", "w");
		if (f == NULL) {
			printf("Error creating mkldnn_result.txt.\n");
			return;
		}

		for(int n=0; n<BATCH_SIZE; n++)
			for(int z=0; z<FC_OUTPUT_SIZE; z++)
				fprintf(f, "%.17g\n", buf_result[z + n * FC_OUTPUT_SIZE]);

		fclose(f);
	}
}

int main()
{
	try {
		conv_relu_fc_softmax();
	}
	catch (error &e) {
		std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
			<< "Error status: " << mkldnn_status2str(e.status) << std::endl;
		return 1;
	}

	return 0;
}
