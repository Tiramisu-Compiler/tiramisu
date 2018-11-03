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
    auto cpu_engine = engine(engine::cpu, 0);
    std::vector<float> net_src(BATCH_SIZE * FIn* (N+K) * (N+K));
    std::vector<float> net_dst(BATCH_SIZE * FOut* N * N);

    /* initializing non-zero values for src */
    srand (1);
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = rand()%10; 

    /*  conv     */
    memory::dims conv_src_tz = { BATCH_SIZE, FIn, (N+K), (N+K) };
    memory::dims conv_weights_tz = { FOut, FIn, K+1, K+1 };
    memory::dims conv_bias_tz = { FOut };
    memory::dims conv_dst_tz = { BATCH_SIZE, FOut, N, N };
    memory::dims conv_strides = { 1, 1 };
    auto conv_padding = { 0,0 };

    std::vector<float> conv_weights(
            std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1,
                            std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
                                                 conv_bias_tz.end(), 1,
                                                 std::multiplies<uint32_t>()));

    /* initializing non-zero values for weights and bias */
    for (int i = 0; i < (int)conv_weights.size(); ++i)
        conv_weights[i] = 1;
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = 0;

    /* create memory for user data */
    auto conv_user_src_memory = memory(
            { { { conv_src_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_src.data());
    auto conv_user_weights_memory
            = memory({ { { conv_weights_tz }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     conv_weights.data());
    auto conv_user_bias_memory = memory(
            { { { conv_bias_tz }, memory::data_type::f32, memory::format::x },
              cpu_engine },
            conv_bias.data());
												        
    /* create mmemory descriptors for convolution data  */
    auto conv_src_md = memory::desc({ conv_src_tz }, memory::data_type::f32,
                                    memory::format::nchw);
    auto conv_bias_md = memory::desc({ conv_bias_tz }, memory::data_type::f32,
                                     memory::format::any);
    auto conv_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::f32, memory::format::nchw);
    auto conv_dst_md = memory::desc({ conv_dst_tz }, memory::data_type::f32,
                                    memory::format::nchw);

    /* create a convolution primitive descriptor */
    auto conv_desc = convolution_forward::desc(
            prop_kind::forward, convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            conv_padding, conv_padding, padding_kind::zero);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);			

    /* create reorder primitives between user input and conv src if needed */
     auto conv_src_memory = conv_user_src_memory;
    bool reorder_conv_src = false;
    primitive conv_reorder_src;
    if (memory::primitive_desc(conv_pd.src_primitive_desc())
        != conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_pd.src_primitive_desc());
        conv_reorder_src = reorder(conv_user_src_memory, conv_src_memory);
        reorder_conv_src = true;
    }

    auto conv_weights_memory = conv_user_weights_memory;
    bool reorder_conv_weights = false;
    primitive conv_reorder_weights;
    if (memory::primitive_desc(conv_pd.weights_primitive_desc())
        != conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_pd.weights_primitive_desc());
        conv_reorder_weights
                = reorder(conv_user_weights_memory, conv_weights_memory);
        reorder_conv_weights = true;
    }


    /* create memory primitive for conv dst */
    auto conv_dst_memory = memory(conv_pd.dst_primitive_desc());

    /* finally create a convolution primitive */
    auto conv = convolution_forward(conv_pd, conv_src_memory, conv_weights_memory,
                                  conv_user_bias_memory, conv_dst_memory);


  /*  relu     */  
    const float negative_slope = 0.0f;

    /* create relu primitive desc */
    /* keep memory format of source same as the format of convolution
     * output in order to avoid reorder */
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv_pd.dst_primitive_desc().desc(),
            negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);

    /* create relu dst memory primitive */
    auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());

    /* finally create a relu primitive */
    auto relu = eltwise_forward(relu_pd, conv_dst_memory, relu_dst_memory);


    /*  conv2     */
    memory::dims conv2_src_tz = { BATCH_SIZE, FOut, N, N};
    memory::dims conv2_weights_tz = { FOut, FOut, K+1, K+1 };
    memory::dims conv2_bias_tz = { FOut };
    memory::dims conv2_dst_tz = { BATCH_SIZE, FOut, N-K, N-K };
    memory::dims conv2_strides = { 1, 1 };
    auto conv2_padding = { 0,0 };

    std::vector<float> conv2_weights(
            std::accumulate(conv2_weights_tz.begin(), conv2_weights_tz.end(), 1,
                            std::multiplies<uint32_t>()));
    std::vector<float> conv2_bias(std::accumulate(conv2_bias_tz.begin(),
                                                 conv2_bias_tz.end(), 1,
                                                 std::multiplies<uint32_t>()));

    /* initializing non-zero values for weights and bias */
    for (int i = 0; i < (int)conv2_weights.size(); ++i)
        conv2_weights[i] = 1;
    for (size_t i = 0; i < conv2_bias.size(); ++i)
        conv2_bias[i] = 0;

    /* create memory for user data */

    auto conv2_user_weights_memory
            = memory({ { { conv2_weights_tz }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     conv2_weights.data());
    auto conv2_user_bias_memory = memory(
            { { { conv2_bias_tz }, memory::data_type::f32, memory::format::x },
              cpu_engine },
            conv2_bias.data());
												        

    auto conv2_src_md = memory::desc({ conv2_src_tz }, memory::data_type::f32,
                                    memory::format::any);
    auto conv2_bias_md = memory::desc({ conv2_bias_tz }, memory::data_type::f32,
                                     memory::format::any);
    auto conv2_weights_md = memory::desc(
            { conv2_weights_tz }, memory::data_type::f32, memory::format::any);
    auto conv2_dst_md = memory::desc({ conv2_dst_tz }, memory::data_type::f32,
                                    memory::format::nchw);

    /* create a convolution primitive descriptor */
    auto conv2_desc = convolution_forward::desc(
            prop_kind::forward, convolution_direct, conv2_src_md,
            conv2_weights_md, conv2_bias_md, conv2_dst_md, conv2_strides,
            conv2_padding, conv2_padding, padding_kind::zero);
    auto conv2_pd = convolution_forward::primitive_desc(conv2_desc, cpu_engine);			

    /* create reorder primitives between user input and conv src if needed */
     auto conv2_src_memory = relu_dst_memory;
    bool reorder_conv2_src = false;
    primitive conv2_reorder_src;
    if (memory::primitive_desc(conv2_pd.src_primitive_desc())
        != relu_dst_memory.get_primitive_desc()) {
        conv2_src_memory = memory(conv2_pd.src_primitive_desc());
        conv2_reorder_src = reorder(relu_dst_memory, conv2_src_memory);
        reorder_conv2_src = true;
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    bool reorder_conv2_weights = false;
    primitive conv2_reorder_weights;
    if (memory::primitive_desc(conv2_pd.weights_primitive_desc())
        != conv2_user_weights_memory.get_primitive_desc()) {
        conv2_weights_memory = memory(conv2_pd.weights_primitive_desc());
        conv2_reorder_weights
                = reorder(conv2_user_weights_memory, conv2_weights_memory);
        reorder_conv2_weights = true;
    }


    /* create memory primitive for conv dst */
    auto conv2_dst_memory = memory(conv2_pd.dst_primitive_desc());

    /* finally create a convolution primitive */
    auto conv2
            = convolution_forward(conv2_pd, conv2_src_memory, conv2_weights_memory,
                                  conv2_user_bias_memory, conv2_dst_memory);


  /*  relu 2    */  
    const float negative_slope2 = 0.0f;

    /* create relu primitive desc */
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv2_pd.dst_primitive_desc().desc(),
            negative_slope2);
    auto relu2_pd = eltwise_forward::primitive_desc(relu2_desc, cpu_engine);

    /* create relu dst memory primitive */
    auto relu2_dst_memory = memory(relu2_pd.dst_primitive_desc());

    /* finally create a relu primitive */
    auto relu2 = eltwise_forward(relu2_pd, conv2_dst_memory, relu2_dst_memory);


 /*  pool  */

    memory::dims pool_dst_tz = { BATCH_SIZE, FOut, N-2*K, N-2*K };
    memory::dims pool_kernel = { K+1, K+1 };
    memory::dims pool_strides = { 1, 1 };
    auto pool_padding = { 0, 0 };

    /* create memory for pool dst data in user format */
    auto pool_user_dst_memory = memory(
            { { { pool_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_dst.data());

    /* create pool dst memory descriptor in format any */
    auto pool_dst_md = memory::desc({ pool_dst_tz }, memory::data_type::f32,
                                    memory::format::any);

    /* create a pooling primitive descriptor */
    auto pool_desc = pooling_forward::desc(
            prop_kind::forward, pooling_max,
            relu2_dst_memory.get_primitive_desc().desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding,
            padding_kind::zero);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, cpu_engine);

    /* create reorder primitive between pool dst and user dst format
     * if needed */
    auto pool_dst_memory = pool_user_dst_memory;
    bool reorder_pool_dst = false;
    primitive pool_reorder_dst;
    if (memory::primitive_desc(pool_pd.dst_primitive_desc())
        != pool_user_dst_memory.get_primitive_desc()) {
        pool_dst_memory = memory(pool_pd.dst_primitive_desc());
        pool_reorder_dst = reorder(pool_dst_memory, pool_user_dst_memory);
        reorder_pool_dst = true;
    }

    /* create pooling workspace memory if training */
    auto pool_workspace_memory = memory(pool_pd.workspace_primitive_desc());

    /* finally create a pooling primitive */
    auto pool = pooling_forward(pool_pd, relu2_dst_memory, pool_dst_memory,
                                pool_workspace_memory);

   
    /* build forward net */
    std::vector<primitive> net_fwd;
    if (reorder_conv_src)
        net_fwd.push_back(conv_reorder_src);
    if (reorder_conv_weights)
        net_fwd.push_back(conv_reorder_weights);
    net_fwd.push_back(conv);
    net_fwd.push_back(relu);
    if (reorder_conv2_src)
        net_fwd.push_back(conv2_reorder_src);
    if (reorder_conv2_weights)
        net_fwd.push_back(conv2_reorder_weights);
    net_fwd.push_back(conv2);
    net_fwd.push_back(relu2);
    net_fwd.push_back(pool);
    if (reorder_pool_dst)
        net_fwd.push_back(pool_reorder_dst);
   
     
    stream(stream::kind::eager).submit(net_fwd).wait();


    printf("writing result in file\n");
    ofstream resultfile;
    resultfile.open ("mkldnn_result.txt");

    float * poolres = (float*)pool_dst_memory.get_data_handle();
    for (size_t i = 0; i < BATCH_SIZE; ++i)
    	for (size_t j = 0; j < FOut; ++j)
		for (size_t k = 0; k < N-2*K; ++k)	
			for (size_t l = 0; l < N-2*K; ++l)	
				resultfile <<poolres[i*FOut*(N-2*K)*(N-2*K) + j*(N-2*K)*(N-2*K) + k*(N-2*K) + l];

    resultfile.close();

}

int main(int argc, char **argv)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;
    try
    {
    	for (int i=0; i<NB_TESTS; i++)
    	{
		auto start1 = std::chrono::high_resolution_clock::now();
		vggBlock();
		auto end1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double,std::milli> duration = end1 - start1;
		duration_vector_2.push_back(duration);
    	}

        std::cout << "\t\tMKL-DNN vggBlock duration" << ": " << median(duration_vector_2) << "; " << std::endl;
    }
    
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
