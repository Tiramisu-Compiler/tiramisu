#include <cstring>
#include <chrono>
#include <iostream>
#include <math.h>
#include <numeric>
#include <fstream>
#include <time.h>
#include <string>
#include <iomanip>
#include "mkldnn.hpp"
#include <assert.h>
#include "configure.h"
#include <cstdlib>

using namespace mkldnn;
using namespace std;

void lstm()
{
        auto cpu_engine = engine(engine::kind::cpu, 0);
        stream s(cpu_engine);
        
        std::vector<primitive> lstm_net;
        std::vector<std::unordered_map<int, memory>> lstm_net_args;
        ofstream performanceFile;
        performanceFile.open("performance_LSTM_CPU_MKLDNN.csv");
        
        int sizes[14][3];

	sizes[0][0] = 64;	sizes[0][1] = 64;	sizes[0][2] = 10;
	sizes[1][0] = 32;	sizes[1][1] = 64;	sizes[1][2] = 10;
	sizes[2][0] = 8;	sizes[2][1] = 64;	sizes[2][2] = 10;
	sizes[3][0] = 16;	sizes[3][1] = 64;	sizes[3][2] = 10;
	sizes[4][0] = 4;	sizes[4][1] = 64;	sizes[4][2] = 10;

	sizes[5][0] = 16;	sizes[5][1] = 128;	sizes[5][2] = 10;
	sizes[6][0] = 16;	sizes[6][1] = 32;	sizes[6][2] = 10;
	sizes[7][0] = 16;	sizes[7][1] = 16;	sizes[7][2] = 10;

	sizes[8][0] = 128;	sizes[8][1] = 16;	sizes[8][2] = 10;
        sizes[9][0] = 64;	sizes[9][1] = 16;	sizes[9][2] = 10;

	sizes[10][0] = 16;	sizes[10][1] = 16;	sizes[10][2] = 20;
	sizes[11][0] = 16;	sizes[11][1] = 16;	sizes[11][2] = 100;
	sizes[12][0] = 16;	sizes[12][1] = 16;	sizes[12][2] = 500;
	sizes[13][0] = 16;	sizes[13][1] = 16;	sizes[13][2] = 1000;

        for (int j = 0; j < 14; j++)
        {
                int F = sizes[j][0];
                int B = sizes[j][1];
                int S = sizes[j][2];
                int N = 4;

                std::vector<float> net_src(S * B * F);
                std::vector<float> net_dst(S * B * F);

                //[Initialize memory]
                memory::dims src_layer_tz = {S, B, F};
                memory::dims weights_layer_tz = {1, 1, F, N, F};
                memory::dims weights_iter_tz = {1, 1, F, N, F};
                memory::dims bias_tz = {1, 1, N, F};
                memory::dims dst_layer_tz = {S, B, F};

                std::vector<float> user_wei_layer(1 * 1 * 2 * F * N * F);
                std::vector<float> user_wei_iter(1 * 1 * F * N * F);
                std::vector<float> user_bias(1 * 1 * N * F);

                srand(0);
                /* Initializing non-zero values for src */
                for (int i = 0; i < (int)net_src.size(); ++i)
                        net_src[i] = (std::rand() % 200) / 100.;

                /* Initializing non-zero values for weights and bias */
                for (int i = 0; i < (int)user_wei_layer.size(); ++i)
                        user_wei_layer[i] = (rand() % 200) / 100.;
                for (int i = 0; i < (int)user_wei_iter.size(); ++i)
                        user_wei_iter[i] = (rand() % 200) / 100.;
                for (int i = 0; i < (int)user_bias.size(); ++i)
                        user_bias[i] = (rand() % 200) / 100.;      
                std::cout << "Initalization done" << std::endl;
        
                // We create the memory descriptors used by the user
                auto user_src_layer_md = mkldnn::memory::desc(
                {src_layer_tz}, mkldnn::memory::data_type::f32,
                mkldnn::memory::format_tag::tnc);

                auto user_wei_layer_md = mkldnn::memory::desc(
                {weights_layer_tz}, mkldnn::memory::data_type::f32,
                mkldnn::memory::format_tag::ldigo);

                auto user_wei_iter_md = mkldnn::memory::desc(
                {weights_iter_tz}, mkldnn::memory::data_type::f32,
                mkldnn::memory::format_tag::ldigo);

                auto user_bias_md = mkldnn::memory::desc(
                {bias_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::ldgo);

                /* auto dst_layer_md = mkldnn::memory::desc(
                {dst_layer_tz}, mkldnn::memory::data_type::f32,
                mkldnn::memory::format_tag::tnc);*/

                /* We create memories */
                auto user_src_layer_memory = mkldnn::memory(user_src_layer_md, cpu_engine,
                                                        net_src.data());
                auto user_wei_layer_memory = mkldnn::memory(user_wei_layer_md, cpu_engine,
                                                        user_wei_layer.data());
                auto user_wei_iter_memory = mkldnn::memory(user_wei_iter_md, cpu_engine,
                                                        user_wei_iter.data());
                auto user_bias_memory = mkldnn::memory(user_bias_md, cpu_engine,
                                                user_bias.data());
        
                //[memory desc for RNN data]
                auto wei_layer_md = memory::desc({ weights_layer_tz },
                        memory::data_type::f32, memory::format_tag::any);

                auto wei_iter_md = memory::desc({ weights_iter_tz },
                        memory::data_type::f32, memory::format_tag::any);

                auto dst_layer_md = memory::desc({ dst_layer_tz },
                        memory::data_type::f32, memory::format_tag::any);
        
                //[create lstm]
                lstm_forward::desc lstm_cell(prop_kind::forward_inference,
                rnn_direction::unidirectional_left2right,
                user_src_layer_md, memory::desc(), memory::desc(),
                user_wei_layer_md, user_wei_iter_md,
                user_bias_md,
                dst_layer_md, memory::desc(), memory::desc());

                auto prim_desc = mkldnn::lstm_forward::primitive_desc(
                lstm_cell, cpu_engine);

                auto wei_layer_memory = memory(prim_desc.weights_layer_desc(), cpu_engine);
                auto wei_layer_reorder_pd = reorder::primitive_desc(user_wei_layer_memory, wei_layer_memory);
                reorder(wei_layer_reorder_pd).execute(s, user_wei_layer_memory, wei_layer_memory);

                auto wei_iter_memory = memory(prim_desc.weights_iter_desc(), cpu_engine);
                auto wei_iter_reorder_pd = reorder::primitive_desc(user_wei_iter_memory, wei_iter_memory);
                reorder(wei_iter_reorder_pd).execute(s, user_wei_iter_memory, wei_iter_memory);

                auto dst_layer_memory = mkldnn::memory(prim_desc.dst_layer_desc(), cpu_engine);

                lstm_net.push_back(lstm_forward(prim_desc));
                lstm_net_args.push_back(
                { { MKLDNN_ARG_SRC_LAYER, user_src_layer_memory },
                        { MKLDNN_ARG_WEIGHTS_LAYER, wei_layer_memory },
                        { MKLDNN_ARG_WEIGHTS_ITER, wei_iter_memory },
                        { MKLDNN_ARG_BIAS, user_bias_memory },
                        { MKLDNN_ARG_DST_LAYER, dst_layer_memory } });

                auto execute = [&]() {
                        assert(lstm_net.size() == lstm_net_args.size()
                                && "something is missing");
                        //[run lstm]
                        for (size_t p = 0; p < lstm_net.size(); ++p)
                        lstm_net.at(p).execute(s, lstm_net_args.at(p));
                };

                std::vector<std::chrono::duration<double, std::milli>> duration_vector;
                for (int i = 0; i < NB_TESTS; i++)
                {
                        auto start1 = std::chrono::high_resolution_clock::now();
                        //stream(stream::kind::eager).submit(lstm_net).wait();
                        execute();
                        s.wait();
                        auto end1 = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> duration = end1 - start1;
                        duration_vector.push_back(duration);
                }

                if (NB_TESTS > 0) {
                        performanceFile << "LSTM; "<< median(duration_vector)<< ";\n";  
                        std::cout << "Kernel \t\t : MKLDNN      ;\n"; 
                        std::cout << "LSTM   \t\t :"<<median(duration_vector)<< "        ;\n"; 

                }

                /* printf("writing result in file\n");
                ofstream resultfile;
                resultfile.open("mkldnn_result.txt");

                float *output = (float *)dst_layer_memory.get_data_handle();
                for (size_t i = 0; i < S; ++i)
                        for (size_t j = 0; j < B; ++j)
                                for (size_t k = 0; k < F; ++k)
                                {
                                        resultfile << output[i * B * F + j * F + k];
                                        resultfile << "\n";
                                }
                resultfile.close();*/
        }
        performanceFile.close();
}

int main(int argc, char **argv)
{
        for (int i = 0; i < NB_TESTS; i++) {
                try
                {     
                        lstm();
                        std::cout << "ok\n";       
                }
                catch (error &e)
                {
                        std::cerr << "status: " << e.status << std::endl;
                        std::cerr << "message: " << e.message << std::endl;
                        return 1;
                }
        }
        return 0;
}