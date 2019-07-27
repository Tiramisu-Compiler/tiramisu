#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include <string>
#include <time.h>
#include "mkldnn.hpp"
#include "configure.h"

using namespace mkldnn;
using namespace std;

void vggBlock()
{
    srand(1);
    std::vector<double> duration_vector;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};

    memory::dims pool_strides = {2, 2};
    memory::dims pool_kernel = {2, 2};
    memory::dims pool_padding = {0, 0};

    std::vector<float> input_buf(BATCH_SIZE*FIn*N*N);

    std::vector<float> conv1_bias_buf(FOut);
    std::vector<float> conv1_weights_buf(FOut*FIn*K*K);

    std::vector<float> conv2_bias_buf(FOut);
    std::vector<float> conv2_weights_buf(FOut*FOut*K*K);

    for (int i = 0; i < BATCH_SIZE*FIn*N*N; i++)
        input_buf[i] = (rand() % 200 - 100) / 100.;

    for (int i = 0; i < FOut; i++)
        conv1_bias_buf[i] = (rand() % 200 - 100) / 100.;
    for (int i = 0; i < FOut*FIn*K*K; i++)
        conv1_weights_buf[i] = (rand() % 200 - 100) / 100.;

    for (int i = 0; i < FOut; i++)
        conv2_bias_buf[i] = (rand() % 200 - 100) / 100.;
    for (int i = 0; i < FOut*FOut*K*K; i++)
        conv2_weights_buf[i] = (rand() % 200 - 100) / 100.;

    // Create memory objects with user data format
    auto input_usr_md = memory::desc(
        {BATCH_SIZE, FIn, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto conv1_weights_usr_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv1_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv1_weights_usr_mem = memory(conv1_weights_usr_md, cpu_engine, conv1_weights_buf.data());
    auto conv1_bias_usr_mem = memory(conv1_bias_usr_md, cpu_engine, conv1_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv1_src_md = memory::desc(
        {BATCH_SIZE, FIn, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv1_weights_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv1_bias_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto output1_md = memory::desc(
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
        conv1_weights_md,
        conv1_bias_md,
        output1_md,
        conv_strides,
        conv_padding,
        conv_padding
    );

    post_ops conv1_post_ops;
    conv1_post_ops.append_eltwise(1, algorithm::eltwise_relu, 0, 0);

    primitive_attr conv1_attr;
    conv1_attr.set_post_ops(conv1_post_ops);

    auto conv1_pd = convolution_forward::primitive_desc(
        conv1_d,
        conv1_attr,
        cpu_engine
    );

    auto conv1_dst_mem = memory(conv1_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto input_usr_mem = memory(input_usr_md, cpu_engine, input_buf.data());
    auto input_mem = memory(conv1_pd.src_desc(), cpu_engine);

    reorder(input_usr_mem, input_mem)
        .execute(cpu_stream, input_usr_mem, input_mem);

    auto conv1_weights_mem = conv1_weights_usr_mem;
    if (conv1_pd.weights_desc() != conv1_weights_usr_mem.get_desc()) {
        conv1_weights_mem = memory(conv1_pd.weights_desc(), cpu_engine);
        reorder(conv1_weights_usr_mem, conv1_weights_mem)
            .execute(cpu_stream, conv1_weights_usr_mem, conv1_weights_mem);
    }

    // Add the first convolution to the network
    net.push_back(convolution_forward(conv1_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_WEIGHTS, conv1_weights_mem},
        {MKLDNN_ARG_BIAS, conv1_bias_usr_mem},
        {MKLDNN_ARG_DST, conv1_dst_mem}
    });

    // Create the second convolution primitive
    auto conv2_weights_usr_md = memory::desc(
        {FOut, FOut, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv2_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv2_weights_usr_mem = memory(conv2_weights_usr_md, cpu_engine, conv2_weights_buf.data());
    auto conv2_bias_usr_mem = memory(conv2_bias_usr_md, cpu_engine, conv2_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv2_weights_md = memory::desc(
        {FOut, FOut, K, K},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv2_bias_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto output2_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv2_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv1_pd.dst_desc(),
        conv2_weights_md,
        conv2_bias_md,
        output2_md,
        conv_strides,
        conv_padding,
        conv_padding
    );

    auto conv2_pd = convolution_forward::primitive_desc(
        conv2_d,
        conv1_attr,
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

    // Add the second convolution to the network
    net.push_back(convolution_forward(conv2_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv1_dst_mem},
        {MKLDNN_ARG_WEIGHTS, conv2_weights_mem},
        {MKLDNN_ARG_BIAS, conv2_bias_usr_mem},
        {MKLDNN_ARG_DST, conv2_dst_mem}
    });

    // Create maxpooling primitive
    auto pool_output_md = memory::desc(
        {BATCH_SIZE, FOut, N/2, N/2},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto pool_d = pooling_forward::desc(
        prop_kind::forward_inference,
        algorithm::pooling_max,
        conv2_pd.dst_desc(),
        pool_output_md,
        pool_strides,
        pool_kernel,
        pool_padding,
        pool_padding
    );

    auto pool_pd = pooling_forward::primitive_desc(
        pool_d,
        cpu_engine
    );

    auto pool_dst_mem = memory(pool_pd.dst_desc(), cpu_engine);

    net.push_back(pooling_forward(pool_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv2_dst_mem},
        {MKLDNN_ARG_DST, pool_dst_mem}
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

    std::cout << "\n\n\tVGG block time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, FOut, N/2, N/2},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto output_mem = memory(output_usr_md, cpu_engine);
    reorder(pool_dst_mem, output_mem)
        .execute(cpu_stream, pool_dst_mem, output_mem);

    /* Write results to file */
    float* output = (float*)output_mem.get_data_handle();
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return ;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N/2; ++y)
                for (int x = 0; x < N/2; ++x)
                    fprintf(f, "%.10g\n", output[x + y*N/2 + fout*N/2*N/2 + n*N/2*N/2*FOut]);

    fclose(f);
}

int main(int argc, char **argv)
{
    try {
		vggBlock();
    }
    
    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
