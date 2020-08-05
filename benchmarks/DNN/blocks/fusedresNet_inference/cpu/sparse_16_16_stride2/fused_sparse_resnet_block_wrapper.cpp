#include "generated_fused_sparse_resnet_block_16_16_stride2.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "configure.h"

// Original version by: Kyle Spafford Adapted for CSR format
void initRandomWeights(float* fin_values, int* filter_idx, int* filter_finptr, const int n, const int KK, const int fin_size, const int fout_size, const int NN, const int stride, int seed)
{
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * fin_size * fout_size;
    double prob = (double)n / ((double) total_num_entries);

    // Seed random number generator
    srand(seed);

    // Randomly decide whether entry gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;

    // We order the weights in the values array such that we do blocking over FIN
    for (int fout = 0; fout < fout_size; fout++)
    {
      filter_finptr[fout] = nnzAssigned;
      for (int fin_b = 0; fin_b < fin_size/FIN_BL; fin_b++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
            for (int ffin = 0; ffin<FIN_BL; ffin++){
              int numEntriesLeft = total_num_entries - ((fout * KK * KK * fin_size) + (fin_b * KK * KK * FIN_BL) + (ky * KK * FIN_BL) + kx * FIN_BL + ffin);
              int needToAssign   = n - nnzAssigned;
              if (numEntriesLeft <= needToAssign) {
                fillRemaining = 1;
              }
              if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
              {
                if (stride == 2)
                  filter_idx[nnzAssigned] = (fin_b * FIN_BL + ffin) * (NN + 2) * (NN + 2) + ky * (NN + 2) + (kx%2) * (NN + 2)/2 + kx/2;
                else
                  filter_idx[nnzAssigned] = (fin_b * FIN_BL + ffin) * (NN + 2) * (NN + 2) + ky * (NN + 2) + kx;

                fin_values[nnzAssigned] = ((float)(rand()%256 - 128)) / 127.f;
                nnzAssigned++;
              }
            }
          }
        }
      }
    }
    filter_finptr[fout_size] = nnzAssigned;
    assert(nnzAssigned == n);
}

int generateCSRWeights(float **filter_values, float density, int **filter_idx, int** filter_finptr, int KK, int fin_size, int fout_size, int NN, int stride, int seed)
{
  int nNonzero = KK * KK * fin_size * fout_size * density;
  *filter_values = (float *) malloc(nNonzero * sizeof(float));
  *filter_idx = (int *) malloc(nNonzero * sizeof(int));
  *filter_finptr = (int *) malloc((fout_size + 1) * sizeof(int));
  initRandomWeights(*filter_values, *filter_idx, *filter_finptr, nNonzero, KK, fin_size, fout_size, NN, stride, seed);
  return nNonzero;
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
        std::cerr << "Couldn't open config file for reading.\n";
    }
}

int main(int, char **)
{
  std::vector<double> duration_vector;
  double start, end;

  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  std::string filename = "resnet_10.csr";
  std::string filename2 = "resnet_10.csr";

  float *filter_values;
  int *filter_idx;
  int *filter_finptr;

  int FNNZ;
  int used_FOUT;
  int used_FIN;
  int used_K;
  int n;
  if (IMPORT_CSR_FROM_FILE)
    importCSRFromFile(filename, &filter_values, &filter_finptr, &filter_idx, &used_FOUT, &used_FIN, &used_K, &FNNZ, &n);
  else{
    used_FOUT = FOut;
    used_FIN = FIn;
    used_K = K;
    n = N;
    FNNZ = generateCSRWeights(&filter_values, WEIGHTS_DENSITY, &filter_idx, &filter_finptr, K, FIn, FOut, N, STRIDE, 2);
  }
  if (IMPORT_CSR_FROM_FILE)
    printf("Layer 1 Density : %.2f %%. Weights imported from %s\n", ((float)FNNZ / (FOut * FIn * K * K))*100, filename.c_str());

  // Assertions to ensure that the generated tiramisu code has the right parameters
  // because we are defining the parameters in the configure.h files to get specialized fast code
  assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
  assert((used_FIN == FIn) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
  assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
  assert((n == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

  Halide::Buffer<int> b_SIZES(2);
  b_SIZES(0) = FNNZ;
  Halide::Buffer<float> b_input((N+2) * (N+2) * FIn, BATCH_SIZE);

  Halide::Buffer<float> b_filter_values(filter_values, FNNZ);
  Halide::Buffer<int> b_filter_idx(filter_idx, FNNZ);
  Halide::Buffer<int> b_filter_finptr(filter_finptr, FOut + 1);

  Halide::Buffer<float> b_bias(FOut);
  Halide::Buffer<float> b_bn_scale(FOut);
  Halide::Buffer<float> b_bn_shift(FOut);
  Halide::Buffer<float> b_bn_mean(FOut);
  Halide::Buffer<float> b_bn_variance(FOut);

  Halide::Buffer<float> b_conv1_result(N/STRIDE + 2, N/STRIDE + 2, FOut, BATCH_SIZE);

  // Second convolution
  float *filter_values2;
  int *filter_idx2;
  int *filter_finptr2;

  int FNNZ2;
  if (IMPORT_CSR_FROM_FILE)
    importCSRFromFile(filename2, &filter_values2, &filter_finptr2, &filter_idx2, &used_FOUT, &used_FIN, &used_K, &FNNZ2, &n);
  else{
    used_FOUT = FOut;
    used_FIN = FOut;
    used_K = K;
    n = N / STRIDE;
    FNNZ2 = generateCSRWeights(&filter_values2, WEIGHTS_DENSITY2, &filter_idx2, &filter_finptr2, K, FOut, FOut, N / STRIDE, 1, 5);
  }
  if (IMPORT_CSR_FROM_FILE)
    printf("Layer 2 Density : %.2f %%. Weights imported from %s\n", ((float)FNNZ2 / (FOut * FOut * K * K))*100, filename2.c_str());
  // Assertions to ensure that the generated tiramisu code has the right parameters
  // because we are defining the parameters in the configure.h files to get specialized fast code
  assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
  assert((used_FIN == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
  assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
  assert((n == N / STRIDE) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

  b_SIZES(1) = FNNZ2;
  Halide::Buffer<float> b_filter_values2(filter_values2, FNNZ2);
  Halide::Buffer<int> b_filter_idx2(filter_idx2, FNNZ2);
  Halide::Buffer<int> b_filter_finptr2(filter_finptr2, FOut + 1);

  Halide::Buffer<float> b_bias2(FOut);
  Halide::Buffer<float> b_bn2_scale(FOut);
  Halide::Buffer<float> b_bn2_shift(FOut);
  Halide::Buffer<float> b_bn2_mean(FOut);
  Halide::Buffer<float> b_bn2_variance(FOut);
  #if PAD_OUTPUT || STRIDE2_FORMATTED_OUTPUT
    Halide::Buffer<float> b_result(N/STRIDE + 2, N/STRIDE + 2, FOut, BATCH_SIZE);
  #else
    Halide::Buffer<float> b_result(N/STRIDE, N/STRIDE, FOut, BATCH_SIZE);
  #endif
  srand(3);
  // First convolution
  if (STRIDE == 2)
    for (int n=0; n < BATCH_SIZE; ++n)
      for (int z=0; z < FIn; ++z)
        for (int y=0; y < N+2; ++y)
          for (int x=0; x < (N+2)/2; ++x)
            for (int xx=0; xx<2; ++xx)
              b_input(x + xx*(N+2)/2 + y * (N + 2) + z* (N + 2) * (N + 2), n) = ((float)(rand()%256 - 128)) / 127.f;
  else
    for (int n=0; n < BATCH_SIZE; ++n)
      for (int z=0; z < FIn; ++z)
        for (int y=0; y < N+2; ++y)
          for (int x=0; x < N+2; ++x)
            b_input(x + y * (N + 2) + z* (N + 2) * (N + 2), n) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<FOut; q++)
    b_bias(q) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<FOut; q++){
    b_bn_scale(q) = 1.f;
    b_bn_shift(q) = 0.f;
    b_bn_mean(q) = ((float)(rand()%256)) / 127.f;
    b_bn_variance(q) = ((float)(rand()%256)) / 127.f;
  }

  // Second convolution
  for (int q=0; q<FOut; q++)
    b_bias2(q) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<FOut; q++){
    b_bn2_scale(q) = 1.f;
    b_bn2_shift(q) = 0.f;
    b_bn2_mean(q) = ((float)(rand()%256)) / 127.f;
    b_bn2_variance(q) = ((float)(rand()%256)) / 127.f;
  }

  std::cout << "Buffers Initialized" << std::endl;
  for (int i = 0; i < NB_TESTS; i++)
  {
    start = rtclock();
		fused_sparse_resnet_block_16_16_stride2(
      b_SIZES.raw_buffer(),
      b_input.raw_buffer(),
        b_filter_values.raw_buffer(),
        b_filter_idx.raw_buffer(),
        b_filter_finptr.raw_buffer(),
        b_bias.raw_buffer(),
        b_bn_scale.raw_buffer(),
        b_bn_shift.raw_buffer(),
        b_bn_mean.raw_buffer(),
        b_bn_variance.raw_buffer(),
      b_conv1_result.raw_buffer(),
        b_filter_values2.raw_buffer(),
        b_filter_idx2.raw_buffer(),
        b_filter_finptr2.raw_buffer(),
        b_bias2.raw_buffer(),
        b_bn2_scale.raw_buffer(),
        b_bn2_shift.raw_buffer(),
        b_bn2_mean.raw_buffer(),
        b_bn2_variance.raw_buffer(),
      b_result.raw_buffer()
    );
    end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  if (SHOW_OUTPUT)
    print_buffer(b_result);

  print_time("performance_CPU.csv", "Sparse ResNet Block",
             {"Tiramisu"},
             {median(duration_vector)});

  if (CHECK_CORRECTNESS){
    // Compare results with Intel MKL
    std::ifstream mkldnn_result("mkl_result.txt");
    double tmp;
    long nb_correct = 0;
    int nbfalse = 0;
    #if STRIDE2_FORMATTED_OUTPUT
      for(int b=0; b<BATCH_SIZE; b++)
        for(int fout=0; fout<FOut; fout++)
          for(int y=0; y<N/STRIDE; y++)
              for(int x=0; x< N/STRIDE; x++){
                mkldnn_result >> tmp;
                if (std::abs(b_result(((x + 1)%2)*((int)((N/STRIDE + 2*PAD_OUTPUT)/2)) + (int)((x + 1)/2), y + 1, fout, b) - tmp) <= 0.002)
                  nb_correct++;
              }
    #else
      for(int b=0; b<BATCH_SIZE; b++)
        for(int fout=0; fout<FOut; fout++)
          for(int y=0; y<N/STRIDE + 2*PAD_OUTPUT; y++)
            for(int x=0; x< N/STRIDE + 2*PAD_OUTPUT; x++){
              #if PAD_OUTPUT
                if((x==0 || y==0 || x==N/STRIDE+1 || y == N/STRIDE+1) && (b_result(x, y, fout, b) == 0)){
                  nb_correct++;
                  continue;
                }
              #endif
              mkldnn_result >> tmp;
              if (std::abs(b_result(x, y, fout, b) - tmp) <= 0.002)
                nb_correct++;
            }
    #endif

    std::cout << "\n\t\tPercentage of correctness " << 100*(((double)nb_correct)/(BATCH_SIZE * FOut * (N/STRIDE + 2*PAD_OUTPUT) * (N/STRIDE + 2*PAD_OUTPUT))) << "%" << std::endl << std::endl;
  }

  free(filter_idx);
  free(filter_values);
  free(filter_finptr);
  return 0;
}
