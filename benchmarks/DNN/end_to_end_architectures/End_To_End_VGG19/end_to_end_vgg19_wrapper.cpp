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
    double start, end;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    std::vector<float> buf1(32*64*(224+2)*(224+2));
    std::vector<float> buf2(32*64*(224+2)*(224+2));
    std::vector<float> buf3(32*64*(224+2)*(224+2));

    for (int i = 0; i < 32*3*226*226;i++)
      buf1[i] = ((float)(rand()%256 - 128)) / 127.f;\
    mkldnn::pooling_forward::primitive_desc dummy_pd;
    // First vgg block
    create_and_add_vgg_block_mkldnn(1,"vgg_weights/vgg_conv_1.csr","vgg_weights/vgg_conv_2.csr",32,64,3,224,3,1,buf1,buf2,buf3, dummy_pd)

    create_and_add_vgg_block_mkldnn(2,"vgg_weights/vgg_conv_3.csr","vgg_weights/vgg_conv_4.csr",32,128,64,112,3,0,buf3,buf2,buf1, pool_pd_1)

    convolution_mkldnn(3,true,"vgg_weights/vgg_conv_5.csr",0.2,32,256,128,56,3,1,0,1,buf1,buf3,true,pool_pd_2)
    convolution_mkldnn(4,true,"vgg_weights/vgg_conv_6.csr",0.2,32,256,256,56,3,1,0,1,buf3,buf1,true,conv_pd_conv_3)
    create_and_add_vgg_block_mkldnn(5,"vgg_weights/vgg_conv_7.csr","vgg_weights/vgg_conv_8.csr",32,256,256,56,3,0,buf1,buf2,buf3, conv_pd_conv_4)

    convolution_mkldnn(6,true,"vgg_weights/vgg_conv_9.csr",0.2,32,512,256,28,3,1,0,1,buf3,buf1,true,pool_pd_5)
    convolution_mkldnn(7,true,"vgg_weights/vgg_conv_10.csr",0.2,32,512,512,28,3,1,0,1,buf1,buf3,true,conv_pd_conv_6)
    create_and_add_vgg_block_mkldnn(8,"vgg_weights/vgg_conv_11.csr","vgg_weights/vgg_conv_12.csr",32,512,512,28,3,0,buf3,buf2,buf1, conv_pd_conv_7)

    convolution_mkldnn(9,true,"vgg_weights/vgg_conv_13.csr",0.2,32,512,512,14,3,1,0,1,buf1,buf3,true,pool_pd_8)
    convolution_mkldnn(10,true,"vgg_weights/vgg_conv_14.csr",0.2,32,512,512,14,3,1,0,1,buf3,buf1,true,conv_pd_conv_9)
    create_and_add_vgg_block_mkldnn(11,"vgg_weights/vgg_conv_15.csr","vgg_weights/vgg_conv_16.csr",32,512,512,14,3,0,buf1,buf2,buf3, conv_pd_conv_10)

    double cutting_point;
    for (int i = 0; i < NB_TESTS; ++i) {
        srand(1);
        for (int i = 0; i < 32*3*226*226;i++)
          buf1[i] = ((float)(rand()%256 - 128)) / 127.f;
        start = rtclock();

        for (size_t j = 0; j < 11; ++j)
            net[j].execute(cpu_stream, net_args[j]);
        cutting_point = rtclock();
        for (size_t j = 11; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);
        // The output is in buf3
        cpu_stream.wait();

        end = rtclock();
        std::cout << i << "th execution : "<< (end - start) * 1000 << " ..... First part time : " << (cutting_point - start) * 1000 << std::endl;
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout<< "size : " << net.size() <<"\t\tVGG : "
    << median(duration_vector) << " ms" << std::endl;


    auto output_usr_md = memory::desc(
      {32, 512, 7, 7},
      memory::data_type::f32,
      memory::format_tag::nchw
    );
    auto result_mem = memory(output_usr_md, cpu_engine);
    reorder(pool_dst_mem_11, result_mem)
      .execute(cpu_stream, pool_dst_mem_11, result_mem);
    float* output = (float*)result_mem.get_data_handle();

    if (WRITE_RESULT_TO_FILE){
      /* Write results to file */
      FILE* f = fopen("mkl_result.txt", "w");
      if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return -3;
      }
      for (int n = 0; n < 32; ++n)
        for (int fout = 0; fout < 512; ++fout)
          for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 7; ++x)
              fprintf(f, "%.17g\n", output[x + y*7 + fout*7*7 + n*7*7*512]);

      fclose(f);
    }

    return 0;
}
