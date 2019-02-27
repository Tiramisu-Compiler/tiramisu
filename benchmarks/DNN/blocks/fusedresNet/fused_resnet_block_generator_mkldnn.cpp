#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include <string>
#include <time.h>
#include <iomanip>
#include "mkldnn.hpp"
#include "configure.h"

using namespace mkldnn;
using namespace std;

void resnetBlock()
{
    auto cpu_engine = engine(engine::cpu, 0);
    std::vector<float> net_src(BATCH_SIZE * 3 * N * N);

    /* Initializing non-zero values for src */
    srand(1);
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = rand() % 1000;

    /****** Conv 1 **********/
    memory::dims conv_src_tz = {BATCH_SIZE, 3, N, N};
    memory::dims conv_weights_tz = {64, 3, 3, 3};
    memory::dims conv_bias_tz = {64};
    memory::dims conv_dst_tz = {BATCH_SIZE, 64, N, N};
    memory::dims conv_strides = {1, 1};
    auto conv_padding = {1, 1};

    std::vector<float> conv_weights(
        std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1,
                        std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
                                                 conv_bias_tz.end(), 1,
                                                 std::multiplies<uint32_t>()));

    /* Initializing non-zero values for weights and bias */
    for (int i = 0; i < (int)conv_weights.size(); ++i)
        conv_weights[i] = 1;
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = 0;

    /* Create memory for user data */
    auto conv_user_src_memory = memory(
        {{{conv_src_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        net_src.data());
    auto conv_user_weights_memory = memory(
        {{{conv_weights_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        conv_weights.data());
    auto conv_user_bias_memory = memory(
        {{{conv_bias_tz}, memory::data_type::f32, memory::format::x},
         cpu_engine},
        conv_bias.data());

    /* Create mmemory descriptors for convolution data  */
    auto conv_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
                                    memory::format::nchw);
    auto conv_bias_md = memory::desc({conv_bias_tz}, memory::data_type::f32,
                                     memory::format::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, memory::data_type::f32,
                                        memory::format::nchw);
    auto conv_dst_md = memory::desc({conv_dst_tz}, memory::data_type::f32,
                                    memory::format::nchw);

    /* Create a convolution primitive descriptor */
    auto conv_desc = convolution_forward::desc(
        prop_kind::forward, convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
        conv_padding, conv_padding, padding_kind::zero);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);

    /* Create reorder primitives between user input and conv src if needed */
    auto conv_src_memory = conv_user_src_memory;
    bool reorder_conv_src = false;
    primitive conv_reorder_src;
    if (memory::primitive_desc(conv_pd.src_primitive_desc()) != conv_user_src_memory.get_primitive_desc())
    {
        conv_src_memory = memory(conv_pd.src_primitive_desc());
        conv_reorder_src = reorder(conv_user_src_memory, conv_src_memory);
        reorder_conv_src = true;
    }

    auto conv_weights_memory = conv_user_weights_memory;
    bool reorder_conv_weights = false;
    primitive conv_reorder_weights;
    if (memory::primitive_desc(conv_pd.weights_primitive_desc()) != conv_user_weights_memory.get_primitive_desc())
    {
        conv_weights_memory = memory(conv_pd.weights_primitive_desc());
        conv_reorder_weights = reorder(conv_user_weights_memory, conv_weights_memory);
        reorder_conv_weights = true;
    }

    /* Create memory primitive for conv dst */
    auto conv_dst_memory = memory(conv_pd.dst_primitive_desc());

    /* Finally create a convolution primitive */
    auto conv = convolution_forward(conv_pd, conv_src_memory, conv_weights_memory,
                                    conv_user_bias_memory, conv_dst_memory);

    /****** Batch normalization 1 **********/
    std::vector<float> bn_dst(BATCH_SIZE * 64 * N * N);
    std::vector<float> mean_vect(64);
    std::vector<float> variance_vect(64);

    memory::dims bn_dst_tz = {BATCH_SIZE, 64, N, N};
    memory::dims mean_tz = {0, 64, 0, 0};
    memory::dims variance_tz = {0, 64, 0, 0};

    /* Create memory for bn dst data in user format */
    auto bn_dst_memory = memory(
        {{{bn_dst_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        bn_dst.data());

    auto mean_memory = memory(
        {{{mean_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        mean_vect.data());

    auto variance_memory = memory(
        {{{variance_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        variance_vect.data());

    /* Create bn dst memory descriptor in format any */
    auto bn_dst_md = memory::desc({bn_dst_tz}, memory::data_type::f32,
                                  memory::format::any);

    /* Create bn primitive descriptor */
    auto epsilon = 0;
    unsigned flags = !mkldnn_use_global_stats && !mkldnn_use_scaleshift && !mkldnn_fuse_bn_relu;

    auto bn_desc = batch_normalization_forward::desc(
        prop_kind::forward, conv_dst_md, epsilon, flags);
    auto bn_pd = batch_normalization_forward::primitive_desc(bn_desc, cpu_engine);

    /* Create a bn primitive */
    auto bn = batch_normalization_forward(bn_pd, conv_dst_memory, bn_dst_memory, mean_memory, variance_memory);

    /****** Relu **********/
    const float negative_slope = 0.0f;

    /* Create relu primitive desc */
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
                                           algorithm::eltwise_relu, bn_pd.dst_primitive_desc().desc(),
                                           negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);

    /* Create relu dst memory primitive */
    auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());

    /* Finally create a relu primitive */
    auto relu = eltwise_forward(relu_pd, bn_dst_memory, relu_dst_memory);

    /****** Conv 2 **********/
    memory::dims conv2_weights_tz = {64, 64, 3, 3};
    memory::dims conv2_bias_tz = {64};
    memory::dims conv2_dst_tz = {BATCH_SIZE, 64, N, N};
    memory::dims conv2_strides = {1, 1};
    auto conv2_padding = {1, 1};

    std::vector<float> conv2_weights(std::accumulate(conv2_weights_tz.begin(),
                                                     conv2_weights_tz.end(), 1,
                                                     std::multiplies<uint32_t>()));
    std::vector<float> conv2_bias(std::accumulate(conv2_bias_tz.begin(),
                                                  conv2_bias_tz.end(), 1,
                                                  std::multiplies<uint32_t>()));

    /* Initializing non-zero values for weights and bias */
    for (int i = 0; i < (int)conv2_weights.size(); ++i)
        conv2_weights[i] = 1;
    for (size_t i = 0; i < conv2_bias.size(); ++i)
        conv2_bias[i] = 0;

    /* Create memory for user data */
    auto conv2_user_weights_memory = memory({{{conv2_weights_tz}, memory::data_type::f32, memory::format::nchw},
                                             cpu_engine},
                                            conv2_weights.data());
    auto conv2_user_bias_memory = memory({{{conv2_bias_tz}, memory::data_type::f32, memory::format::x},
                                          cpu_engine},
                                         conv2_bias.data());

    /* Create mmemory descriptors for convolution data  */
    auto conv2_bias_md = memory::desc({conv2_bias_tz}, memory::data_type::f32,
                                      memory::format::any);
    auto conv2_weights_md = memory::desc({conv2_weights_tz}, memory::data_type::f32,
                                         memory::format::nchw);
    auto conv2_dst_md = memory::desc({conv2_dst_tz}, memory::data_type::f32,
                                     memory::format::nchw);

    /* Create a convolution primitive descriptor */
    auto conv2_desc = convolution_forward::desc(
        prop_kind::forward, convolution_direct, bn_dst_md,
        conv2_weights_md, conv2_bias_md, conv2_dst_md, conv2_strides,
        conv2_padding, conv2_padding, padding_kind::zero);
    auto conv2_pd = convolution_forward::primitive_desc(conv2_desc, cpu_engine);

    /* Create reorder primitives between user input and conv src if needed */
    auto conv2_src_memory = relu_dst_memory;
    bool reorder_conv2_src = false;
    primitive conv2_reorder_src;
    if (memory::primitive_desc(conv2_pd.src_primitive_desc()) != relu_dst_memory.get_primitive_desc())
    {
        conv2_src_memory = memory(conv2_pd.src_primitive_desc());
        conv2_reorder_src = reorder(relu_dst_memory, conv2_src_memory);
        reorder_conv2_src = true;
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    bool reorder_conv2_weights = false;
    primitive conv2_reorder_weights;
    if (memory::primitive_desc(conv2_pd.weights_primitive_desc()) != conv2_user_weights_memory.get_primitive_desc())
    {
        conv2_weights_memory = memory(conv2_pd.weights_primitive_desc());
        conv2_reorder_weights = reorder(conv2_user_weights_memory, conv2_weights_memory);
        reorder_conv2_weights = true;
    }

    /* Create memory primitive for conv dst */
    auto conv2_dst_memory = memory(conv2_pd.dst_primitive_desc());

    /* Finally create a convolution primitive */
    auto conv2 = convolution_forward(conv2_pd, conv2_src_memory, conv2_weights_memory,
                                     conv2_user_bias_memory, conv2_dst_memory);

    /****** Batch normalization 2 **********/
    std::vector<float> bn2_dst(BATCH_SIZE * 64 * N * N);
    std::vector<float> mean2_vect(64);
    std::vector<float> variance2_vect(64);

    memory::dims bn2_dst_tz = {BATCH_SIZE, 64, N, N};
    memory::dims mean2_tz = {0, 64, 0, 0};
    memory::dims variance2_tz = {0, 64, 0, 0};

    /* Create memory for bn dst data in user format */
    auto bn2_dst_memory = memory(
        {{{bn2_dst_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        bn2_dst.data());

    auto mean2_memory = memory(
        {{{mean2_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        mean2_vect.data());

    auto variance2_memory = memory(
        {{{variance2_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        variance2_vect.data());

    /* Create bn2 dst memory descriptor in format any */
    auto bn2_dst_md = memory::desc({bn2_dst_tz}, memory::data_type::f32,
                                   memory::format::any);

    auto bn2_desc = batch_normalization_forward::desc(
        prop_kind::forward, conv2_dst_md, epsilon, flags);
    auto bn2_pd = batch_normalization_forward::primitive_desc(bn2_desc, cpu_engine);

    /* Create a bn primitive */
    auto bn2 = batch_normalization_forward(bn2_pd, conv2_dst_memory, bn2_dst_memory, mean2_memory, variance2_memory);

    /* Build forward net */
    std::vector<primitive> net_fwd;
    if (reorder_conv_src)
        net_fwd.push_back(conv_reorder_src);
    if (reorder_conv_weights)
        net_fwd.push_back(conv_reorder_weights);
    net_fwd.push_back(conv);
    net_fwd.push_back(bn);
    net_fwd.push_back(relu);
    if (reorder_conv2_src)
        net_fwd.push_back(conv2_reorder_src);
    if (reorder_conv2_weights)
        net_fwd.push_back(conv2_reorder_weights);
    net_fwd.push_back(conv2);
    net_fwd.push_back(bn2);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;
    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        stream(stream::kind::eager).submit(net_fwd).wait();
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }
    std::cout << "\t\tMKL-DNN convolution duration"
              << ": " << median(duration_vector_2) << "; " << std::endl;

    printf("writing result in file\n");
    ofstream resultfile;
    resultfile.open("mkldnn_result.txt");

    float *output = (float *)bn2_dst_memory.get_data_handle();
    for (size_t i = 0; i < BATCH_SIZE; ++i)
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < N; ++k)
                for (size_t l = 0; l < N; ++l)
                {
                    resultfile << fixed << setprecision(2) << (float)((int)(output[i * 64 * N * N + j * N * N + k * N + l] * 1000) / 1000.0);
                    resultfile << "\n";
                }
    resultfile.close();
}

int main(int argc, char **argv)
{
    try
    {
        resnetBlock();
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
