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

void densenet_block()
{
    srand(1);
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};

    std::vector<float> input_buf(BATCH_SIZE*4*GR*N*N);
    std::vector<float> bn_buf(BATCH_SIZE*4*GR*N*N);

    std::vector<float> bn_scale_shift_buf(2*4*GR);
    std::vector<float> conv_weights_buf(GR*4*GR*K_Y*K_X);
    std::vector<float> conv_bias_buf(GR);

    for (int z = 0; z < 4*GR; ++z) {
        bn_scale_shift_buf[z] = ((float)(rand()%256)) / 255.f;
        if (bn_scale_shift_buf[z] == 0.f)
            bn_scale_shift_buf[z] = 1.f;

        bn_scale_shift_buf[z + 4*GR] = ((float)(rand()%256 - 128)) / 127.f;
    }

    for (int fout = 0; fout < GR; ++fout)
        for (int z = 0; z < 4*GR; ++z)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_weights_buf[k_x + k_y*K_X + z*K_X*K_Y + fout*K_X*K_Y*4*GR] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*4*GR] = ((float)(rand()%256 - 128)) / 127.f;

    // Create convolution primitive
    // We start by creating convolution in order to get its data layout

    // Create memory objects with user data format
    auto conv_weights_usr_md = memory::desc(
        {GR, 4*GR, K_Y, K_X},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv_bias_usr_md = memory::desc(
        {GR},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv_weights_buf.data());
    auto conv_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv_src_md = memory::desc(
        {BATCH_SIZE, 4*GR, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_weights_md = memory::desc(
        {GR, 4*GR, K_Y, K_X},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_bias_md = memory::desc(
        {GR},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto output_md = memory::desc(
        {BATCH_SIZE, GR, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv_src_md,
        conv_weights_md,
        conv_bias_md,
        output_md,
        conv_strides,
        conv_padding,
        conv_padding
    );

    auto conv_pd = convolution_forward::primitive_desc(
        conv_d,
        cpu_engine
    );

    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);

    // Create BN fused with ReLU primitive

    // Create memory objects first
    auto bn_scale_md = memory::desc(
        {2, 4*GR},
        memory::data_type::f32,
        memory::format_tag::nc
    );

    auto input_usr_md = memory::desc(
        {BATCH_SIZE, 4*GR, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto bn_scale_mem = memory(bn_scale_md, cpu_engine, bn_scale_shift_buf.data());
    auto input_usr_mem = memory(input_usr_md, cpu_engine, input_buf.data());
    auto bn_mem = memory(conv_pd.src_desc(), cpu_engine);

    // Create the primitive
    auto bn_d = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        conv_pd.src_desc(),
        EPSILON,
        batch_normalization_flags::use_scale_shift | batch_normalization_flags::fuse_bn_relu
    );

    auto bn_pd = batch_normalization_forward::primitive_desc(
        bn_d, cpu_engine
    );

    // Edit user data format
    auto input_mem = memory(conv_pd.src_desc(), cpu_engine);
    reorder(input_usr_mem, input_mem)
        .execute(cpu_stream, input_usr_mem, input_mem);

    auto conv_weights_mem = conv_weights_usr_mem;
    if (conv_pd.weights_desc() != conv_weights_usr_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), cpu_engine);
        reorder(conv_weights_usr_mem, conv_weights_mem)
            .execute(cpu_stream, conv_weights_usr_mem, conv_weights_mem);
    }

    // Create the network
    net.push_back(batch_normalization_forward(bn_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_SCALE_SHIFT, bn_scale_mem},
        {MKLDNN_ARG_DST, bn_mem}
    });

    net.push_back(convolution_forward(conv_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, bn_mem},
        {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
        {MKLDNN_ARG_BIAS, conv_bias_usr_mem},
        {MKLDNN_ARG_DST, conv_dst_mem}
    });

    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t j = 0; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);

        cpu_stream.wait();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        duration_vector.push_back(duration);
    }

    std::cout << "\n\n\tDenseNet block time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, GR, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto output_mem = memory(output_usr_md, cpu_engine);
    reorder(conv_dst_mem, output_mem)
        .execute(cpu_stream, conv_dst_mem, output_mem);

    /* Write results to file */
    float* output = (float*)output_mem.get_data_handle();
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return ;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < GR; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output[x + y*N + fout*N*N + n*N*N*GR]);

    fclose(f);
}

int main()
{
    try {
        densenet_block();
    }

    catch (error &e) {
        std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
                  << "Error status: " << mkldnn_status2str(e.status) << std::endl;

        return 1;
    }

    return 0;
}
