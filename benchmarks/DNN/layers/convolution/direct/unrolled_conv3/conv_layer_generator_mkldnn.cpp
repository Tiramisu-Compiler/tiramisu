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

void conv()
{
    srand(1);
    std::vector<double> duration_vector;
    double start, end;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {0, 0};

    std::vector<float> input_buf(BATCH_SIZE*FIn*(N + 2)*(N + 2));

    std::vector<float> conv_bias_buf(FOut);
    std::vector<float> conv_weights_buf(FOut*FIn*K*K);

    for (int i = 0; i < BATCH_SIZE*FIn*(N + 2)*(N + 2); i++)
        input_buf[i] = ((float) (rand()%256 - 128)) / 127.f;

    for (int i = 0; i < FOut*FIn*K*K; i++)
        conv_weights_buf[i] = ((float) (rand()%256 - 128)) / 127.f;

    for (int i = 0; i < FOut; i++)
        conv_bias_buf[i] = ((float) (rand()%256 - 128)) / 127.f;

    // Create memory objects with user data format
    auto input_usr_md = memory::desc(
        {BATCH_SIZE, FIn, N + 2, N + 2},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto conv_weights_usr_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv_weights_buf.data());
    auto conv_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv_src_md = memory::desc(
        {BATCH_SIZE, FIn, N + 2, N + 2},
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
        {BATCH_SIZE, FOut, N, N},
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

    // Edit user data format
    auto input_usr_mem = memory(input_usr_md, cpu_engine, input_buf.data());
    auto input_mem = memory(conv_pd.src_desc(), cpu_engine);

    reorder(input_usr_mem, input_mem)
        .execute(cpu_stream, input_usr_mem, input_mem);

    auto conv_weights_mem = conv_weights_usr_mem;
    if (conv_pd.weights_desc() != conv_weights_usr_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), cpu_engine);
        reorder(conv_weights_usr_mem, conv_weights_mem)
            .execute(cpu_stream, conv_weights_usr_mem, conv_weights_mem);
    }

    // Add convolution to the network
    net.push_back(convolution_forward(conv_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
        {MKLDNN_ARG_BIAS, conv_bias_usr_mem},
        {MKLDNN_ARG_DST, conv_dst_mem}
    });
 
    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        start = rtclock();
        
        for (size_t j = 0; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);

        cpu_stream.wait();

        end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\n\n\tConv time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
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
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output[x + y*N + fout*N*N + n*N*N*FOut]);

    fclose(f);
}

int main(int argc, char **argv)
{
    try {
		conv();
    }
    
    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
