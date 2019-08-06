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

void transpose_conv()
{
    	int OUTPUT_N = ((N-1)*STRIDE + K - 2 * PADDING);
    	std::vector<double> duration_vector;
	
    	engine cpu_engine(engine::kind::cpu, 0);
    	stream cpu_stream(cpu_engine);
	
    	primitive deconv;
    	std::unordered_map<int, memory>deconv_args;
	
    	//Initialize buffers
    	memory::dims strides = {STRIDE, STRIDE};
    	memory::dims paddings = {PADDING, PADDING};
	
    	std::vector<float> buf_input_batch(BATCH_SIZE*FIn*N*N);
    	std::vector<float> buf_filter(FOut*FIn*K*K);
    	std::vector<float> buf_bias(FOut);
	
    	for (int oc = 0; oc < FOut; oc++)
		for (int ic = 0; ic < FIn; ic++)
			for (int ky = 0; ky < K; ky++)
				for (int kx = 0; kx < K; kx++)
					buf_filter[kx + ky*K + ic*K*K + oc*K*K*FIn] = 1.0f;
	
    	for (int oc = 0; oc < FOut; oc++)
		buf_bias[oc] = 1.0f;
	
    	for (int b = 0; b < BATCH_SIZE; b++)
		for (int ic = 0; ic < FIn; ic++)
			for (int y = 0; y < N; y++)
				for (int x = 0; x < N; x++)
					buf_input_batch[x + y*N + ic*N*N + b*N*N*FIn] = 1.0f;
	
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
	
    	// Create memory objects with a data format selected by the deconvolution primitive
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
	
    	// Create the deconvolution primitive descriptor
    	auto transposed_conv_desc = deconvolution_forward::desc(
		prop_kind::forward_inference,
		algorithm::deconvolution_direct,
		conv_src_md,
		conv_weights_md,
		conv_bias_md,
		output_md,
		strides,
		paddings,
		paddings
	);
	
    	auto transposed_conv_prim_desc = deconvolution_forward::primitive_desc(transposed_conv_desc, cpu_engine);
	
    	auto input_user_md = memory::desc(
		{BATCH_SIZE, FIn, N, N},
		memory::data_type::f32,
		memory::format_tag::nchw
    	);
	
    	auto conv_dst_memory = memory(transposed_conv_prim_desc.dst_desc(), cpu_engine);
    	auto input_user_memory = memory(input_user_md, cpu_engine, buf_input_batch.data());
	
    	auto input_memory = memory(transposed_conv_prim_desc.src_desc(), cpu_engine);
	
    	//Convert the user's input to the primitive's format
    	reorder(input_user_memory, input_memory)
	.execute(cpu_stream, input_user_memory, input_memory);
	
    	auto conv_weights_memory = conv_weights_user_memory;
    	if (transposed_conv_prim_desc.weights_desc() != conv_weights_user_memory.get_desc()) 
	{
		conv_weights_memory = memory(transposed_conv_prim_desc.weights_desc(), cpu_engine);
		reorder(conv_weights_user_memory, conv_weights_memory)
			.execute(cpu_stream, conv_weights_user_memory, conv_weights_memory);
	}
	
    	/* create deconvolution primitive */
    	deconv = deconvolution_forward(transposed_conv_prim_desc);
    	deconv_args={
		{MKLDNN_ARG_SRC, input_memory},
		{MKLDNN_ARG_WEIGHTS, conv_weights_memory},
		{MKLDNN_ARG_BIAS, conv_bias_user_memory},
		{MKLDNN_ARG_DST, conv_dst_memory}
	};
	
    	// Execute the deconvolution
	for (int i = 0; i < NB_TESTS; ++i) {
		double start = rtclock();

		deconv.execute(cpu_stream, deconv_args);

		cpu_stream.wait();

		double end = rtclock();
		duration_vector.push_back((end - start) * 1000);
	}
	
    	std::cout << "\n\n\tMKLDNN Deconv time : " << median(duration_vector) << " ms." << std::endl;
	
    	// Convert convolution output to user data format
	auto output_user_md = memory::desc(
		{BATCH_SIZE, FOut, OUTPUT_N, OUTPUT_N},
		memory::data_type::f32,
		memory::format_tag::nchw
	);
	
	auto output_memory = memory(output_user_md, cpu_engine);
	reorder(conv_dst_memory, output_memory)
	.execute(cpu_stream, conv_dst_memory, output_memory);
	
	float* output = (float*)output_memory.get_data_handle();
	
	if (SHOW_OUTPUT)
	{
		for (int n = 0; n < BATCH_SIZE; n++)
			for (int fout = 0; fout < FOut; fout++){
				for (int y = 0; y < OUTPUT_N; y++){
					for (int x = 0; x < OUTPUT_N; x++)
						printf("%.10g, ", output[x + y*OUTPUT_N + fout*OUTPUT_N*OUTPUT_N + n*OUTPUT_N*OUTPUT_N*FOut]);
					printf("\n");
				}
				printf("\n");
			}
	}
	
	if (SAVE_TO_FILE_AND_COMPARE)
	{
		// Write results to file
		FILE* f = fopen("mkldnn_result.txt", "w");
		if (f == NULL) {
			printf("Error creating mkl_result.txt.\n");
			exit(2);
		}
		
		for (int n = 0; n < BATCH_SIZE; n++)
			for (int z = 0; z < FOut; z++)
				for (int y = 0; y < OUTPUT_N; y++)
					for (int x = 0; x < OUTPUT_N; x++)
						fprintf(f, "%.17g\n", output[x + y*OUTPUT_N + z*OUTPUT_N*OUTPUT_N + n*OUTPUT_N*OUTPUT_N*FOut]);
		
		fclose(f);
	}
}

int main()
{
	try {
		transpose_conv();
	}
	catch (error &e) {
		std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
		<< "Error status: " << mkldnn_status2str(e.status) << std::endl;
		return 1;
	}
	
	return 0;
}
