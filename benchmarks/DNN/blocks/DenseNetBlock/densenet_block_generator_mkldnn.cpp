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

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};

    std::vector<float> input_buf(BATCH_SIZE*4*GR*N*N);
    std::vector<float> conv_weights_buf(GR*4*GR*K_Y*K_X);
    std::vector<float> conv_bias_buf(GR);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*4*GR] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        for (int z = 0; z < 4*GR; ++z)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_weights_buf[k_x + k_y*K_X + z*K_X*K_Y + fout*K_X*K_Y*4*GR] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    // Create convolution primitive

    // Create memory objects with user data format
    auto input_md = memory::desc(
        {BATCH_SIZE, 4*GR, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

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

    auto input_mem = memory(input_md, cpu_engine, input_buf.data());
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

    // Edit user data format if needed
    auto conv_src_mem = input_mem;
    if (conv_pd.src_desc() != input_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), cpu_engine);

        reorder(input_mem, conv_src_mem)
            .execute(cpu_stream, input_mem, conv_src_mem);
    }

    auto conv_weights_mem = conv_weights_usr_mem;
    if (conv_pd.weights_desc() != conv_weights_usr_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), cpu_engine);
        reorder(conv_weights_usr_mem, conv_weights_mem)
            .execute(cpu_stream, conv_weights_usr_mem, conv_weights_mem);
    }

    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);

    auto conv = convolution_forward(conv_pd);

    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        conv.execute(cpu_stream, {
            {MKLDNN_ARG_SRC, conv_src_mem},
            {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
            {MKLDNN_ARG_BIAS, conv_bias_usr_mem},
            {MKLDNN_ARG_DST, conv_dst_mem}
        });

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
