#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <fstream>
#include "mkl.h"
#include "mkldnn.hpp"

#include "configure.h"
#include "mkldnn_creation_macros.h"
using namespace mkldnn;

void importCSRFromFileAsDense(std::string filename, float** matrix, int* FOUT, int* FIN, int* KK, int* n){
    std::ifstream cFile (filename);
    float* values;
    int* rowptr;
    int* colidx;
    int NNZ;

    if (cFile.is_open())
    {
        std::string line;
        // Get first line containing conv size

        getline(cFile, line);
        std::string delimiter = ",";

        size_t pos = 0;
        std::string token;
        // FOUT
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        *FOUT = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        // FIN
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        *FIN = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        // K
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        *KK = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        // NNZ
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        *n = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        // NNZ
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        NNZ = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        values = (float*)malloc((NNZ) * sizeof(float));
        rowptr = (int*)malloc(((*FOUT) + 1) * sizeof(int));
        colidx = (int*)malloc((NNZ) * sizeof(int));
        int i = 0;
        getline(cFile, line);
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            values[i] = std::stof(line);
            i++;
        }
        assert(i == NNZ);

        i = 0;
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            rowptr[i] = std::stoi(line);
            i++;
        }
        assert(i == (*FOUT + 1));

        i = 0;
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            colidx[i] = std::stoi(line);
            i++;
        }
        assert(i == NNZ);

        // Transform to dense
        *matrix = (float*)malloc(((*FOUT) * (*FIN) * (*KK) * (*KK)) * sizeof(float));
        memset(*matrix, 0.f, ((*FOUT) * (*FIN) * (*KK) * (*KK)) * sizeof(float));
        for (int fout = 0; fout < *FOUT; fout++){
          int fstart = rowptr[fout];
          int fend = rowptr[fout + 1];
          for(int i = fstart; i < fend; i++){
            int fin = colidx[i] / (*n + 2) / (*n + 2);
            int ky = colidx[i] / (*n + 2) % (*n + 2);
            int kx = colidx[i] % (*n + 2);

            (*matrix)[fout * (*FIN) * (*KK) * (*KK) + fin * (*KK) * (*KK) + ky * (*KK) + kx] = values[i];
          }
        }
        free(values);
        free(rowptr);
        free(colidx);
    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }
}

int main()
{
    std::vector<double> duration_vector;

    srand(1);

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net, net_relu;
    std::vector<std::unordered_map<int, memory>> net_args, net_args_relu;
    std::vector<float> input_buf(32*32*(224+2)*(224+2));
    std::vector<float> output_buf(32*32*(224+2)*(224+2));
    std::vector<float> conv1x1_buf(32*32*226*226);

    for (int i = 0; i < 32*3*226*226;i++)
      input_buf[i] = ((float)(rand()%256 - 128)) / 127.f;\

    // 3x3 Conv-ReLU-MaxPool
    convolution_mkldnn(1,true,"resnet_weights/resnet_1.csr",0.2,32,16,3,224,3,1,0,1,input_buf,output_buf,true,input_usr_md_conv_1)
    maxpool_mkldnn(1,32,16,224,output_buf,input_buf,conv_pd_conv_1,conv_dst_mem_conv_1)

    // First block 16-16 with stride = 2 for the first convolution
    create_and_add_resnet_block_mkldnn(2,"resnet_weights/resnet_2.csr","resnet_weights/resnet_3.csr",2,32,16,16,112,3,input_buf,output_buf,pool_pd_maxpool_1)
    convolution_mkldnn(2,false,"",0.2,32,16,16,112,1,2,0,0,input_buf,conv1x1_buf,false,pool_pd_maxpool_1.dst_desc())
    create_relu_mkldnn(2,output_buf,bn2_pd_2)

    // Second block 16-16 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(3,"resnet_weights/resnet_4.csr","resnet_weights/resnet_5.csr",1,32,16,16,56,3,output_buf,input_buf,bn2_pd_2)
    create_relu_mkldnn(3,input_buf,bn2_pd_3)

    // Third block 16-16 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(4,"resnet_weights/resnet_6.csr","resnet_weights/resnet_7.csr",1,32,16,16,56,3,input_buf,output_buf,bn2_pd_3)
    create_relu_mkldnn(4,output_buf,bn2_pd_4)

    // Fourth block 16-32 with stride = 2 for the first convolution
    create_and_add_resnet_block_mkldnn(5,"resnet_weights/resnet_8.csr","resnet_weights/resnet_9.csr",2,32,32,16,56,3,output_buf,input_buf,bn2_pd_4)
    convolution_mkldnn(5,false,"",0.2,32,32,16,56,1,2,0,0,output_buf,conv1x1_buf,false,bn2_pd_4.dst_desc())
    create_relu_mkldnn(5,input_buf,bn2_pd_5)

    // Fifth block 32-32 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(6,"resnet_weights/resnet_10.csr","resnet_weights/resnet_11.csr",1,32,32,32,28,3,input_buf,output_buf,bn2_pd_5)
    create_relu_mkldnn(6,output_buf,bn2_pd_6)

    // Sixth block 32-32 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(7,"resnet_weights/resnet_12.csr","resnet_weights/resnet_13.csr",1,32,32,32,28,3,output_buf,input_buf,bn2_pd_6)
    create_relu_mkldnn(7,input_buf,bn2_pd_7)

    // Seventh block 32-64 with stride = 2 for the first convolution
    create_and_add_resnet_block_mkldnn(8,"resnet_weights/resnet_14.csr","resnet_weights/resnet_15.csr",2,32,64,32,28,3,input_buf,output_buf,bn2_pd_7)
    convolution_mkldnn(8,false,"",0.2,32,64,32,28,1,2,0,0,input_buf,conv1x1_buf,false,bn2_pd_7.dst_desc())
    create_relu_mkldnn(8,output_buf,bn2_pd_8)

    // Eighth block 64-64 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(9,"resnet_weights/resnet_16.csr","resnet_weights/resnet_17.csr",1,32,64,64,14,3,output_buf,input_buf,bn2_pd_8)
    create_relu_mkldnn(9,input_buf,bn2_pd_9)

    // Ninth block 64-64 with stride = 1 for both convolutions
    create_and_add_resnet_block_mkldnn(10,"resnet_weights/resnet_18.csr","resnet_weights/resnet_19.csr",1,32,64,64,14,3,input_buf,output_buf,bn2_pd_9)
    create_relu_mkldnn(10,output_buf,bn2_pd_10)

    double start, end;
    double conv_relu_maxpool_time, conv_time, resnet_time;
    double resnet_block_2, resnet_block_3,resnet_block_4, resnet_block_5, resnet_block_6, resnet_block_7, resnet_block_8, resnet_block_9;
    double add_relu;
    for (int i = 0; i < NB_TESTS; ++i) {
        srand(1);
        for (int i = 0; i < 32*3*226*226;i++)
          input_buf[i] = ((float)(rand()%256 - 128)) / 127.f;
        start = rtclock();

        // 3x3 Conv-ReLU-MaxPool
        for (size_t j = 0; j < 2; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        conv_relu_maxpool_time = rtclock();

//////////////////////////////// 16 to 16 part
        // First block 16-16 with first conv stride = 2
        for (size_t j = 2; j < 6; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        resnet_time = rtclock();
        reorder(input_usr_mem_conv_2, input_mem_conv_2)\
            .execute(cpu_stream, input_usr_mem_conv_2, input_mem_conv_2);
        CONV1x1_THEN_ADD(6,0,conv1x1_buf,output_buf,32*16*56*56)
        conv_time = rtclock();

        // Second block
        for (size_t j = 7; j < 11; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        add_relu = rtclock();
        NAIVE_ADD_RELU(1,input_buf,output_buf,32*16*56*56)
        resnet_block_2 = rtclock();

        // Third block
        for (size_t j = 11; j < 15; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        NAIVE_ADD_RELU(2, output_buf,input_buf,32*16*56*56)
        resnet_block_3 = rtclock();

/////////////////////////////////
        // Fourth block 16-32 with first conv stride = 2
        for (size_t j = 15; j < 19; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        reorder(input_usr_mem_conv_5, input_mem_conv_5)
            .execute(cpu_stream, input_usr_mem_conv_5, input_mem_conv_5);
        CONV1x1_THEN_ADD(19,3,conv1x1_buf,input_buf,32*32*28*28)
        resnet_block_4 = rtclock();

//////////////////////////////// 32 to 32 part
        // Fifth block
        for (size_t j = 20; j < 24; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        NAIVE_ADD_RELU(4, output_buf,input_buf,32*32*28*28)
        resnet_block_5 = rtclock();

        // Sixth block
        for (size_t j = 24; j < 28; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        NAIVE_ADD_RELU(5, input_buf,output_buf,32*32*28*28)
        resnet_block_6 = rtclock();

//////////////////////////////////
        // Seventh block 32-64 with first conv stride = 2
        for (size_t j = 28; j < 32; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        reorder(input_usr_mem_conv_8, input_mem_conv_8)
            .execute(cpu_stream, input_usr_mem_conv_8, input_mem_conv_8);
        CONV1x1_THEN_ADD(32,6,conv1x1_buf,output_buf,32*64*14*14)
        resnet_block_7 = rtclock();

///////////////////////////////// 64 to 64 part
        // Eighth block
        for (size_t j = 33; j < 37; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        NAIVE_ADD_RELU(7, input_buf,output_buf,32*64*14*14)
        resnet_block_8 = rtclock();

        // Ninth block
        for (size_t j = 37; j < 41; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        NAIVE_ADD_RELU(8, output_buf,input_buf,32*64*14*14)
        resnet_block_9 = rtclock();

        // The output is in output_buf
        cpu_stream.wait();

        end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout<< "size : " << net.size() <<"\t\tResNet : "
    << median(duration_vector) << " ms" << std::endl;

    if (DEBUG_PER_BLOCK_TIME){
      printf("Time CRM %lf\n", (conv_relu_maxpool_time - start)*1000);
      printf("S2 Time ResNetBlock %lf\n", (conv_time - conv_relu_maxpool_time)*1000);
      printf("\t\t BLOCK : %lf\n", (resnet_time - conv_relu_maxpool_time)*1000);
      printf("\t\t Convolution : %lf\n", (conv_time - resnet_time)*1000);
      printf("Time ResNetBlock2 %lf\n", (resnet_block_2 - conv_time)*1000);
      printf("\t\t BLOCK : %lf\n", (add_relu - conv_time)*1000);
      printf("\t\t Add-ReLU : %lf\n", (resnet_block_2 - add_relu)*1000);

      printf("Time ResNetBlock3 %lf\n", (resnet_block_3 - resnet_block_2)*1000);
      printf("S2 Time ResNetBlock4 %lf\n", (resnet_block_4 - resnet_block_3)*1000);
      printf("Time ResNetBlock5 %lf\n", (resnet_block_5 - resnet_block_4)*1000);
      printf("Time ResNetBlock6 %lf\n", (resnet_block_6 - resnet_block_5)*1000);
      printf("S2 Time ResNetBlock7 %lf\n", (resnet_block_7 - resnet_block_6)*1000);
      printf("Time ResNetBlock8 %lf\n", (resnet_block_8 - resnet_block_7)*1000);
      printf("Time ResNetBlock9 %lf\n", (resnet_block_9 - resnet_block_8)*1000);
    }

    auto output_usr_md = memory::desc(
      {32, 64, 14, 14},
      memory::data_type::f32,
      memory::format_tag::nchw
    );
    auto result_mem = memory(output_usr_md, cpu_engine);
    reorder(conv2_dst_mem_10, result_mem)
      .execute(cpu_stream, conv2_dst_mem_10, result_mem);
    float* output = (float*)result_mem.get_data_handle();

    if (WRITE_RESULT_TO_FILE){
      /* Write results to file */
      FILE* f = fopen("mkl_result.txt", "w");
      if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return -3;
      }
      for (int n = 0; n < 32; ++n)
        for (int fout = 0; fout < 64; ++fout)
          for (int y = 0; y < 14; ++y)
            for (int x = 0; x < 14; ++x)
              fprintf(f, "%.17g\n", output[x + y*14 + fout*14*14 + n*14*14*64]);

      fclose(f);
    }

    return 0;
}
