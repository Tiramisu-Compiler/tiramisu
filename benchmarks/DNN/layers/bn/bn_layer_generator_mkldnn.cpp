#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include <string>
#include <time.h>
#include "mkldnn.hpp"
#include "configure.h"
#include <iomanip>

using namespace mkldnn;
using namespace std;

void bn_mkldnn()
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector;
    auto cpu_engine = engine(engine::cpu, 0);
    std::vector<float> net_src(BATCH_SIZE * FIn * N * N);
    std::vector<float> net_dst(BATCH_SIZE * FIn * N * N);
    std::vector<float> mean_vect(FIn);
    std::vector<float> variance_vect(FIn);

    /* Initializing non-zero values for src */
    srand(1);
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = rand() % 100;

    memory::dims src_tz = {BATCH_SIZE, FIn, N, N};
    memory::dims dst_tz = {BATCH_SIZE, FIn, N, N};
    memory::dims mean_tz = {0, FIn, 0, 0};
    memory::dims variance_tz = {0, FIn, 0, 0};

    /* Create memory for user data */
    auto user_src_memory = memory(
        {{{src_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        net_src.data());

    /* Create memory for bn dst data in user format */
    auto dst_memory = memory(
        {{{dst_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        net_dst.data());

    auto mean_memory = memory(
        {{{mean_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        mean_vect.data());

    auto variance_memory = memory(
        {{{variance_tz}, memory::data_type::f32, memory::format::nchw},
         cpu_engine},
        variance_vect.data());

    /* Create mmemory descriptors for source data  */
    auto src_md = memory::desc({src_tz}, memory::data_type::f32,
                               memory::format::nchw);

    /* Create bn dst memory descriptor in format any */
    auto bn_dst_md = memory::desc({dst_tz}, memory::data_type::f32,
                                  memory::format::any);

    /* Create bn primitive descriptor */
    unsigned flags = !mkldnn_use_global_stats && !mkldnn_use_scaleshift && !mkldnn_fuse_bn_relu;

    auto bn_desc = batch_normalization_forward::desc(
        prop_kind::forward, src_md, EPSILON, flags);
    auto bn_pd = batch_normalization_forward::primitive_desc(bn_desc, cpu_engine);

    /* Create a bn primitive */
    auto bn = batch_normalization_forward(bn_pd, user_src_memory, dst_memory, mean_memory, variance_memory);

    /* Build forward net */
    std::vector<primitive> net_fwd;
    net_fwd.push_back(bn);

    for (int i = 0; i < NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        stream(stream::kind::eager).submit(net_fwd).wait();
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end1 - start1;
        duration_vector.push_back(duration);
    }

    std::cout << "\t\tMKL-DNN BN duration"
              << ": " << median(duration_vector) << "; " << std::endl;

    ofstream resultfile;
    resultfile.open("mkldnn_result.txt");

    float *bnres = (float *)dst_memory.get_data_handle();
    for (size_t i = 0; i < BATCH_SIZE; ++i)
        for (size_t j = 0; j < FIn; ++j)
            for (size_t k = 0; k < N; ++k)
                for (size_t l = 0; l < N; ++l)
                    resultfile << (float)((int)(bnres[i * FIn * N * N + j * N * N + k * N + l] * 100) / 100.0);

    resultfile.close();
}

int main(int argc, char **argv)
{
    try
    {
        bn_mkldnn();
    }

    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
