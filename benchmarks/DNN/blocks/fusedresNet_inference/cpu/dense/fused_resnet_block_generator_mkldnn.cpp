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

void resnet_block()
{
    srand(1);
    std::vector<double> duration_vector;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    std::vector<float> bn_scale_shift_buf(2*FOut);
    std::vector<float> bn_mean_buf(FOut);
    std::vector<float> bn_variance_buf(FOut);
    std::vector<float> bn2_scale_shift_buf(2*FOut);
    std::vector<float> bn2_mean_buf(FOut);
    std::vector<float> bn2_variance_buf(FOut);

    std::vector<float> input_buf(BATCH_SIZE*FIn*(N+2)*(N+2));

    std::vector<float> conv1_weights_buf(FOut*FIn*K_Y*K_X);
    std::vector<float> conv1_bias_buf(FOut);
    memory::dims conv1_padding = {0, 0};

    std::vector<float> conv2_weights_buf(FOut*FIn*K_Y*K_X);
    std::vector<float> conv2_bias_buf(FOut);
    memory::dims conv2_padding = {1, 1};

    for (int fout = 0; fout < FOut; ++fout) {
        bn_scale_shift_buf[fout] = 1.f;
        bn_scale_shift_buf[fout + FOut] = 0.f;

        bn_mean_buf[fout] = ((float)(rand()%256)) / 127.f;
        bn_variance_buf[fout] = ((float)(rand()%256)) / 127.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv1_weights_buf[k_x + k_y*K_X + fin*K_X*K_Y + fout*K_X*K_Y*FIn] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        conv1_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout) {
        bn2_scale_shift_buf[fout] = 1.f;
        bn2_scale_shift_buf[fout + FOut] = 0.f;

        bn2_mean_buf[fout] = ((float)(rand()%256)) / 127.f;
        bn2_variance_buf[fout] = ((float)(rand()%256)) / 127.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv2_weights_buf[k_x + k_y*K_X + fin*K_X*K_Y + fout*K_X*K_Y*FIn] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        conv2_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < FIn; ++fin)
            for (int y = 0; y < N + 2; ++y)
                for (int x = 0; x < N + 2; ++x)
                    input_buf[x + y*(N+2) + fin*(N+2)*(N+2) + n*(N+2)*(N+2)*FIn] = ((float)(rand()%256 - 128)) / 127.f;

    // Create first convolution primitive

    // Create memory objects with user data format
    auto input_usr_md = memory::desc(
        {BATCH_SIZE, FIn, N+2, N+2},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto conv_weights_usr_md = memory::desc(
        {FOut, FIn, K_Y, K_X},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv1_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv1_weights_buf.data());
    auto conv1_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv1_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv1_src_md = memory::desc(
        {BATCH_SIZE, FIn, N+2, N+2},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_weights_md = memory::desc(
        {FOut, FIn, K_Y, K_X},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_bias_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_output_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv1_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv1_src_md,
        conv_weights_md,
        conv_bias_md,
        conv_output_md,
        conv_strides,
        conv1_padding,
        conv1_padding
    );

    auto conv1_pd = convolution_forward::primitive_desc(
        conv1_d,
        cpu_engine
    );

    auto conv1_dst_mem = memory(conv1_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto input_mem = memory(conv1_pd.src_desc(), cpu_engine);
    auto user_input_mem = memory(input_usr_md, cpu_engine, input_buf.data());

    reorder(user_input_mem, input_mem)
        .execute(cpu_stream, user_input_mem, input_mem);

    auto conv1_weights_mem = conv1_weights_usr_mem;
    if (conv1_pd.weights_desc() != conv1_weights_usr_mem.get_desc()) {
        conv1_weights_mem = memory(conv1_pd.weights_desc(), cpu_engine);
        reorder(conv1_weights_usr_mem, conv1_weights_mem)
            .execute(cpu_stream, conv1_weights_usr_mem, conv1_weights_mem);
    }

    net.push_back(convolution_forward(conv1_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_WEIGHTS, conv1_weights_mem},
        {MKLDNN_ARG_BIAS, conv1_bias_usr_mem},
        {MKLDNN_ARG_DST, conv1_dst_mem}
    });

    // Create BN fused with ReLU primitive
    auto bn_scale_md = memory::desc(
        {2, FOut},
        memory::data_type::f32,
        memory::format_tag::nc
    );

    auto bn_mean_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto bn_variance_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto bn1_scale_mem = memory(bn_scale_md, cpu_engine, bn_scale_shift_buf.data());
    auto bn1_mean_mem = memory(bn_mean_md, cpu_engine, bn_mean_buf.data());
    auto bn1_variance_mem = memory(bn_variance_md, cpu_engine, bn_variance_buf.data());

    auto bn1_d = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        conv1_pd.dst_desc(),
        EPSILON,
        mkldnn::normalization_flags::use_scale_shift | mkldnn::normalization_flags::fuse_norm_relu | mkldnn::normalization_flags::use_global_stats
    );

    auto bn1_pd = batch_normalization_forward::primitive_desc(
        bn1_d, cpu_engine
    );

    net.push_back(batch_normalization_forward(bn1_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv1_dst_mem},
        {MKLDNN_ARG_SCALE_SHIFT, bn1_scale_mem},
        {MKLDNN_ARG_MEAN, bn1_mean_mem},
        {MKLDNN_ARG_VARIANCE, bn1_variance_mem},
        {MKLDNN_ARG_DST, conv1_dst_mem}
    });

    // Create second convolution primitive

    // Create memory objects with user data format
    auto conv2_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv2_weights_buf.data());
    auto conv2_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv2_bias_buf.data());

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv2_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv1_pd.dst_desc(),
        conv_weights_md,
        conv_bias_md,
        conv_output_md,
        conv_strides,
        conv2_padding,
        conv2_padding
    );

    auto conv2_pd = convolution_forward::primitive_desc(
        conv2_d,
        cpu_engine
    );

    auto conv2_dst_mem = memory(conv2_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto conv2_weights_mem = conv2_weights_usr_mem;
    if (conv2_pd.weights_desc() != conv2_weights_usr_mem.get_desc()) {
        conv2_weights_mem = memory(conv2_pd.weights_desc(), cpu_engine);
        reorder(conv2_weights_usr_mem, conv2_weights_mem)
            .execute(cpu_stream, conv2_weights_usr_mem, conv2_weights_mem);
    }

    net.push_back(convolution_forward(conv2_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv1_dst_mem},
        {MKLDNN_ARG_WEIGHTS, conv2_weights_mem},
        {MKLDNN_ARG_BIAS, conv2_bias_usr_mem},
        {MKLDNN_ARG_DST, conv2_dst_mem}
    });

    // Create BN fused with ReLU primitive
    auto bn2_scale_mem = memory(bn_scale_md, cpu_engine, bn2_scale_shift_buf.data());
    auto bn2_mean_mem = memory(bn_mean_md, cpu_engine, bn2_mean_buf.data());
    auto bn2_variance_mem = memory(bn_variance_md, cpu_engine, bn2_variance_buf.data());

    auto bn2_d = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        conv2_pd.dst_desc(),
        EPSILON,
        mkldnn::normalization_flags::use_scale_shift | mkldnn::normalization_flags::use_global_stats
    );

    auto bn2_pd = batch_normalization_forward::primitive_desc(
        bn2_d, cpu_engine
    );

    net.push_back(batch_normalization_forward(bn2_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv2_dst_mem},
        {MKLDNN_ARG_SCALE_SHIFT, bn2_scale_mem},
        {MKLDNN_ARG_MEAN, bn2_mean_mem},
        {MKLDNN_ARG_VARIANCE, bn2_variance_mem},
        {MKLDNN_ARG_DST, conv2_dst_mem}
    });

    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();
        
        for (size_t j = 0; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);

        cpu_stream.wait();

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\n\n\tResNet block time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto output_mem = memory(output_usr_md, cpu_engine);
    reorder(conv2_dst_mem, output_mem)
        .execute(cpu_stream, conv2_dst_mem, output_mem);

    /* Write results to file */
    float* output = (float*)output_mem.get_data_handle();
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return ;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output[x + y*N + fout*N*N + n*N*N*FOut]);

    fclose(f);
}

int main()
{
    try {
        resnet_block();
    }

    catch (error &e) {
        std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
                  << "Error status: " << mkldnn_status2str(e.status) << std::endl;

        return 1;
    }

    return 0;
}