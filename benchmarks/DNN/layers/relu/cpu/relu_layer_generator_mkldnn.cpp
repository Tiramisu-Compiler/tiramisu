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

void relu()
{
    auto cpu_engine = engine(engine::cpu, 0);
    std::vector<float> net_src(BATCH_SIZE * FIn * N * N);
    std::vector<float> net_dst(BATCH_SIZE * FIn * N * N);

    /* initializing non-zero values for src */
    srand(1);
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = rand() % 10 - 5;

    memory::dims src_tz = {BATCH_SIZE, FIn, N, N};

    /* create memory for user data */
    auto user_src_memory = memory(
        {{{src_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        net_src.data());

    auto src_md = memory::desc({src_tz}, memory::data_type::f32,
                               memory::format::nchw);

    /* relu     */
    const float negative_slope = NEGATIVE_SLOPES;

    /* create relu primitive desc */
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
                                           algorithm::eltwise_relu, src_md, negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);

    /* create relu dst memory primitive */
    auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());

    /* finally create a relu primitive */
    auto relu = eltwise_forward(relu_pd, user_src_memory, relu_dst_memory);

    /* build forward net */
    std::vector<primitive> net_fwd;
    net_fwd.push_back(relu);

    std::vector<std::chrono::duration<double, std::milli>> duration_vector_2;
    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        stream(stream::kind::eager).submit(net_fwd).wait();
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector_2.push_back(duration);
    }
    std::cout << "\t\tMKL-DNN relu duration"
              << ": " << median(duration_vector_2) << "; " << std::endl;

    ofstream resultfile;
    resultfile.open("mkldnn_result.txt");

    float *relures = (float *)relu_dst_memory.get_data_handle();
    for (size_t i = 0; i < BATCH_SIZE; ++i)
        for (size_t j = 0; j < FIn; ++j)
            for (size_t k = 0; k < N; ++k)
                for (size_t l = 0; l < N; ++l)
                {
                    resultfile << relures[i * FIn * N * N + j * N * N + k * N + l];
                    resultfile << "\n";
                }
    resultfile.close();
}

int main(int argc, char **argv)
{
    try
    {
        relu();
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
