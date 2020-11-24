#include "tiramisu_functions/generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o.h"
#include "tiramisu_functions/generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o.h"

#include "tiramisu_functions/generated_sparse_vgg_block_512_512_28_tiramisu.o.h"
#include "tiramisu_functions/generated_sparse_vgg_block_512_512_14_tiramisu.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include "mkldnn.hpp"

#include "configure.h"
#include "tiramisu_creation_macros.h"
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
void importCSRFromFile(std::string filename, float** values, int** rowptr, int** colidx, int* FOUT, int* FIN, int* KK, int* NNZ, int* n){
    std::ifstream cFile (filename);

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
        *NNZ = std::stoi(token);
        line.erase(0, pos + delimiter.length());

        *values = (float*)malloc((*NNZ) * sizeof(float));
        *rowptr = (int*)malloc(((*FOUT) + 1) * sizeof(int));
        *colidx = (int*)malloc((*NNZ) * sizeof(int));
        int i = 0;
        getline(cFile, line);
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            (*values)[i] = std::stof(line);
            i++;
        }
        assert(i == *NNZ);

        i = 0;
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            (*rowptr)[i] = std::stoi(line);
            i++;
        }
        assert(i == (*FOUT + 1));

        i = 0;
        while(getline(cFile, line)){
            if(line[0]=='/' || line.empty())
              break;
            (*colidx)[i] = std::stoi(line);
            i++;
        }
        assert(i == *NNZ);
    }
    else {
        std::cerr << "Couldn't open file for reading.\n";
    }
}

int main(int, char **)
{
  std::vector<double> duration_vector;
  double start, end;

  engine cpu_engine(engine::kind::cpu, 0);
  stream cpu_stream(cpu_engine);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  std::vector<float> buf1(32*64*(224+2)*(224+2));
  std::vector<float> buf2(32*64*(224+2)*(224+2));
  std::vector<float> buf3(32*64*226*226);

  srand(1);
  for (int i = 0; i < 32*3*226*226;i++)
    buf1[i] = ((float)(rand()%256 - 128)) / 127.f;

  mkldnn::pooling_forward::primitive_desc dummy_pd;

  // Creating the dense part using MKL DNN
  create_and_add_vgg_block_mkldnn(1,"vgg_weights/vgg_conv_1.csr","vgg_weights/vgg_conv_2.csr",32,64,3,224,3,1,buf1,buf2,buf3, dummy_pd)

  create_and_add_vgg_block_mkldnn(2,"vgg_weights/vgg_conv_3.csr","vgg_weights/vgg_conv_4.csr",32,128,64,112,3,0,buf3,buf2,buf1, pool_pd_1)

  convolution_mkldnn(3,true,"vgg_weights/vgg_conv_5.csr",0.2,32,256,128,56,3,1,0,1,buf1,buf3,true,pool_pd_2)
  convolution_mkldnn(4,true,"vgg_weights/vgg_conv_6.csr",0.2,32,256,256,56,3,1,0,1,buf3,buf1,true,conv_pd_conv_3)
  create_and_add_vgg_block_mkldnn(5,"vgg_weights/vgg_conv_7.csr","vgg_weights/vgg_conv_8.csr",32,256,256,56,3,0,buf1,buf2,buf3, conv_pd_conv_4)

  auto unpadded_output_tiramisu_md = memory::desc(
      {32, 256, 28, 28},
      memory::data_type::f32,
      memory::format_tag::nchw
  );

  auto output_mem = memory(unpadded_output_tiramisu_md, cpu_engine, buf1.data());

  // Creating the Sparse part using Tiramisu Sparse
  create_sparse_conv_relu_conv_relu_tiramisu(6,"vgg_weights/vgg_conv_9.csr","vgg_weights/vgg_conv_10.csr",32,512,256,28,3,buf3.data(),buf1.data())
  create_sparse_vgg_block_tiramisu(7,"vgg_weights/vgg_conv_11.csr","vgg_weights/vgg_conv_12.csr",32,512,512,28,3,buf1.data(),buf3.data())

  create_sparse_conv_relu_conv_relu_tiramisu(8,"vgg_weights/vgg_conv_13.csr","vgg_weights/vgg_conv_14.csr",32,512,512,14,3,buf3.data(),buf1.data())
  create_sparse_vgg_block_tiramisu(9,"vgg_weights/vgg_conv_15.csr","vgg_weights/vgg_conv_16.csr",32,512,512,14,3,buf1.data(),buf3.data())

  std::cout << "Architecture Initialized: "<< net.size() << std::endl;

  for (int i = 0; i < NB_TESTS; i++)
  {
    srand(1);
    for (int i = 0; i < 32*3*226*226;i++)
      buf1[i] = ((float)(rand()%256 - 128)) / 127.f;
    start = rtclock();

    // Calling the dense part
    for (size_t j = 0; j < net.size(); ++j)
        net[j].execute(cpu_stream, net_args[j]);
    reorder(pool_dst_mem_5, output_mem)
        .execute(cpu_stream, pool_dst_mem_5, output_mem);
    memset(buf3.data(), 0, 32*256*(30)*(30));
    // Transforming the dense part output data layout so it can be given as an input to the sparse part
    for (int b=0; b<32; b++)
      for(int fout=0; fout<256; fout++)
        for(int y=0; y<28; y++)
          for(int x=0; x<28; x++)
            buf3[x + 1 + (y + 1)*30 + fout*30*30 + b*30*30*256] = buf1[x + y*28 + fout*28*28 + b*28*28*256];

    cpu_stream.wait();

    // Calling the sparse part
    call_sparse_conv_relu_conv_relu_tiramisu(6,_256_512_28_tiramisu)
    call_sparse_vgg19_block_tiramisu(7,_512_512_28_tiramisu)
    call_sparse_conv_relu_conv_relu_tiramisu(8,_512_512_14_tiramisu)
    call_sparse_vgg19_block_tiramisu(9,_512_512_14_tiramisu)

    end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  print_time("performance_CPU.csv", "Sparse VGG19",
             {"Tiramisu"},
             {median(duration_vector)});

  Halide::Buffer<float> b_result(buf3.data(), 7 + 2, 7 + 2, 512, 32);

  if (WRITE_RESULT_TO_FILE){
   // Write results to file
   FILE* f = fopen("tiramisu_result.txt", "w");
   if (f == NULL) {
     printf("Error creating tiramisu_result.txt.\n");
     return 0;
   }

   for(int b=0; b<32; b++)
     for(int fout=0; fout<512; fout++)
       for(int y=0; y<7 + 2; y++)
         for(int x=0; x< 7 + 2; x++)
           fprintf(f, "%.17g\n", b_result(x, y, fout, b));

   fclose(f);
  }

  if (CHECK_CORRECTNESS){
   // Compare results with Intel MKL-DNN
   std::ifstream mkldnn_result("mkl_result.txt");
   double tmp;
   long nb_correct = 0;

   for(int b=0; b<32; b++)
     for(int fout=0; fout<512; fout++)
       for(int y=0; y<7 + 2; y++)
         for(int x=0; x< 7 + 2; x++){
           if((x==0 || y==0 || x==7+1 || y == 7+1) && (b_result(x, y, fout, b) == 0)){
             nb_correct++;
             continue;
           }
           mkldnn_result >> tmp;
           if (abs(b_result(x, y, fout, b) - tmp) <= 0.1)
             nb_correct++;
         }
   std::cout << "\t\tPercentage of correctness " << 100*(((double)nb_correct)/(32 * 512 * (7 + 2) * (7 + 2))) << "%" << std::endl << std::endl;
  }
  return 0;
}
