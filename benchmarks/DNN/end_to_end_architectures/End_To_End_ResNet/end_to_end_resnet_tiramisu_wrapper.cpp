#include "tiramisu_functions/generated_spconv_relu_maxpool.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block_16_16_stride2.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block16_16.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block_16_32_stride2.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block32_32.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block_32_64_stride2.o.h"
#include "tiramisu_functions/generated_fused_sparse_resnet_block64_64.o.h"

#include "tiramisu_functions/add_relu/add_relu_inplace_32_16_58_tiramisu.o.h"
#include "tiramisu_functions/add_relu/add_relu_inplace_32_32_30_tiramisu.o.h"
#include "tiramisu_functions/add_relu/add_relu_inplace_32_64_16_tiramisu.o.h"

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

  std::vector<float> buf1(32*32*(224+2)*(224+2));
  std::vector<float> buf2(32*32*(224+2)*(224+2));
  std::vector<float> buf3(32*32*226*226);

  srand(1);
  for (int i = 0; i < 32*3*226*226;i++)
    buf1[i] = ((float)(rand()%256 - 128)) / 127.f;

  create_conv_relu_maxpool_3_16(1,"resnet_weights/resnet_1.csr",32,16,3,224,3,buf1.data(),buf2.data())

  // First part
  // Transform buf2 -> buf1
  create_sparse_resnet_block_tiramisu(2,"resnet_weights/resnet_stride2_2.csr","resnet_weights/resnet_3.csr",32,16,16,112,3,2,buf1.data(),buf3.data())
  convolution_mkldnn_tiramisu_format(2,false,"",0.2,32,16,16,112,1,2,1,1,buf2,buf1,buf3,58,false)//output in buf3

  create_sparse_resnet_block_tiramisu(3,"resnet_weights/resnet_4.csr","resnet_weights/resnet_5.csr",32,16,16,56,3,1,buf3.data(),buf1.data())
  create_add_relu_block_tiramisu(3,32,16,58,buf1.data(),buf3.data())

  create_sparse_resnet_block_tiramisu(4,"resnet_weights/resnet_6.csr","resnet_weights/resnet_7.csr",32,16,16,56,3,1,buf1.data(),buf3.data())
  create_add_relu_block_tiramisu(4,32,16,58,buf3.data(),buf1.data())

  // Second part
  create_sparse_resnet_block_tiramisu(5,"resnet_weights/resnet_8.csr","resnet_weights/resnet_9.csr",32,32,16,56,3,2,buf3.data(),buf2.data())
  convolution_mkldnn_tiramisu_format(5,false,"",0.2,32,32,16,56,1,2,1,1,buf3,buf1,buf2,30,false)//output in buf2

  create_sparse_resnet_block_tiramisu(6,"resnet_weights/resnet_10.csr","resnet_weights/resnet_11.csr",32,32,32,28,3,1,buf2.data(),buf1.data())
  create_add_relu_block_tiramisu(6,32,32,30,buf1.data(),buf2.data())

  create_sparse_resnet_block_tiramisu(7,"resnet_weights/resnet_12.csr","resnet_weights/resnet_13.csr",32,32,32,28,3,1,buf1.data(),buf2.data())
  create_add_relu_block_tiramisu(7,32,32,30,buf2.data(),buf1.data())

  // Third part
  create_sparse_resnet_block_tiramisu(8,"resnet_weights/resnet_14.csr","resnet_weights/resnet_15.csr",32,64,32,28,3,2,buf2.data(),buf3.data())
  convolution_mkldnn_tiramisu_format(8,false,"",0.2,32,64,32,28,1,2,1,1,buf2,buf1,buf3,16,false)

  create_sparse_resnet_block_tiramisu(9,"resnet_weights/resnet_16.csr","resnet_weights/resnet_17.csr",32,64,64,14,3,1,buf3.data(),buf2.data())
  create_add_relu_block_tiramisu(9,32,64,16,buf2.data(),buf3.data())

  create_sparse_resnet_block_tiramisu(10,"resnet_weights/resnet_18.csr","resnet_weights/resnet_19.csr",32,64,64,14,3,1,buf2.data(),buf3.data())
  create_add_relu_block_tiramisu(10,32,64,16,buf3.data(),buf2.data())


  std::cout << "Buffers Initialized" << std::endl;
  double conv_relu_maxpool_time, conv_time, resnet_time;
  double resnet_block_2, resnet_block_3,resnet_block_4, resnet_block_5, resnet_block_6, resnet_block_7, resnet_block_8, resnet_block_9;
  double add_relu;

  for (int i = 0; i < NB_TESTS; i++)
  {
    srand(1);
    for (int i = 0; i < 32*3*226*226;i++)
      buf1[i] = ((float)(rand()%256 - 128)) / 127.f;
    start = rtclock();
    // Conv ReLU MaxPool

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_spconv_relu_maxpool_tiramisu(1)
    conv_relu_maxpool_time = rtclock();
    // First part
    // Transform buf2->buf1
    #pragma omp parallel for
    for(int b = 0; b<32; b++)
      for(int fin = 0; fin<16; fin++)
        for(int y = 0; y<112 + 2; y++)
          #pragma omp simd
          for(int x = 0; x<112 + 2; x++)
            buf1[b*114*114*16 + fin*114*114 + y*114 + (x%2)*(114)/2 + x/2] = buf2[b*114*114*16 + fin*114*114 + y*114 + x];
    call_sparse_resnet_block_tiramisu(2,_16_16_stride2)
    resnet_time = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    CONV1x1_THEN_ADD_RELU_TIRAMISU(2,0,32,16,58)
    conv_time = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(3,16_16)
    add_relu = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(3, 32, 16, 58)
    resnet_block_2 = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(4,16_16)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(4, 32, 16, 58)

    resnet_block_3 = rtclock();

    // Second part

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(5,_16_32_stride2)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    CONV1x1_THEN_ADD_RELU_TIRAMISU(5,1,32,32,30)
    resnet_block_4 = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(6,32_32)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(6, 32, 32, 30)
    resnet_block_5 = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(7,32_32)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(7, 32, 32, 30)
    resnet_block_6 = rtclock();

    // Third part

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(8,_32_64_stride2)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    CONV1x1_THEN_ADD_RELU_TIRAMISU(8,2,32,64,16)
    resnet_block_7 = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(9,64_64)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(9, 32, 64, 16)
    resnet_block_8 = rtclock();

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=8");
    #endif
    call_sparse_resnet_block_tiramisu(10,64_64)

    #if NUM_THREADS_TUNING
      putenv("HL_NUM_THREADS=4");
    #endif
    ADD_RELU_TIRAMISU(10, 32, 64, 16)
    resnet_block_9 = rtclock();

    cpu_stream.wait();
    end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  print_time("performance_CPU.csv", "Sparse ResNet Block",
             {"Tiramisu"},
             {median(duration_vector)});
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

  Halide::Buffer<float> b_result(buf3.data(), 14 + 2, 14 + 2, 64, 32);

  if (WRITE_RESULT_TO_FILE){
   // Write results to file
   FILE* f = fopen("tiramisu_result.txt", "w");
   if (f == NULL) {
     printf("Error creating tiramisu_result.txt.\n");
     return 0;
   }

   for(int b=0; b<32; b++)
     for(int fout=0; fout<64; fout++)
       for(int y=0; y<14 + 2; y++)
         for(int x=0; x< 14 + 2; x++)
           fprintf(f, "%.17g\n", b_result(x, y, fout, b));

   fclose(f);
  }

  if (CHECK_CORRECTNESS){
   // Compare results with Intel MKL
   std::ifstream mkldnn_result("mkl_result.txt");
   double tmp;
   long nb_correct = 0;

   for(int b=0; b<32; b++)
     for(int fout=0; fout<64; fout++)
       for(int y=0; y<14 + 2; y++)
         for(int x=0; x< 14 + 2; x++){
           if((x==0 || y==0 || x==14+1 || y == 14+1) && (b_result(x, y, fout, b) == 0)){
             nb_correct++;
             continue;
           }
           mkldnn_result >> tmp;
           if (abs(b_result(x, y, fout, b) - tmp) <= 0.0001)
             nb_correct++;
         }
   std::cout << "\t\tPercentage of correctness " << 100*(((double)nb_correct)/(32 * 64 * (14 + 2) * (14 + 2))) << "%" << std::endl << std::endl;
  }
  return 0;
}
