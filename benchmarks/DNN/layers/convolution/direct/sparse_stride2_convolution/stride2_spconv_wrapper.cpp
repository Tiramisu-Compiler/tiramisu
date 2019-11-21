#include "generated_stride2_spconv.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "configure.h"
// Original version by: Kyle Spafford Adapted for CSR format
void initRandomWeights(float* fin_values, int* filter_idx, int* filter_finptr, const int n, const int KK, const int fin_size, const int fout_size, int seed)
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
                filter_idx[nnzAssigned] = (fin_b * FIN_BL + ffin) * (N + 2) * (N + 2) + ky * (N + 2) + (kx%2) * (N + 2)/2 + kx/2;
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

int generateCSRWeights(float **filter_values, float density, int **filter_idx, int** filter_finptr, int KK, int fin_size, int fout_size, int seed)
{
  int nNonzero = KK * KK * fin_size * fout_size * density;
  *filter_values = (float *) malloc(nNonzero * sizeof(float));
  *filter_idx = (int *) malloc(nNonzero * sizeof(int));
  *filter_finptr = (int *) malloc((fout_size + 1) * sizeof(int));
  initRandomWeights(*filter_values, *filter_idx, *filter_finptr, nNonzero, KK, fin_size, fout_size, seed);
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
        //printf(" DENSITY : %f\n", (float)(*NNZ)/ ((float)(*FOUT) * (*FIN) * (*KK) * (*KK)));
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
  std:: string filename = "resnet_10.csr";

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
    FNNZ = generateCSRWeights(&filter_values, WEIGHTS_DENSITY, &filter_idx, &filter_finptr, used_K, used_FIN, used_FOUT, 3);
  }

  // Assertions to ensure that the generated tiramisu code has the right parameters
  // because we are defining the parameters in the configure.h files to get specialized fast code
  assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
  assert((used_FIN == FIn) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
  assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
  assert((n == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

  Halide::Buffer<int> b_SIZES(1);
  b_SIZES(0) = FNNZ;
  Halide::Buffer<float> b_input((N+2) * (N+2) * used_FIN, BATCH_SIZE);

  Halide::Buffer<float> b_filter_values(filter_values, FNNZ);
  Halide::Buffer<int> b_filter_idx(filter_idx, FNNZ);
  Halide::Buffer<int> b_filter_finptr(filter_finptr, used_FOUT + 1);

  Halide::Buffer<float> b_bias(used_FOUT);

  Halide::Buffer<float> b_result(N/2, N/2, used_FOUT, BATCH_SIZE);

  srand(2);
  for (int n=0; n < BATCH_SIZE; ++n)
    for (int z=0; z < used_FIN; ++z)
      for (int y=0; y < N+2; ++y)
        for (int x=0; x < (N+2)/2; ++x)
          for (int xx=0; xx<2; ++xx)
            b_input(x + xx*(N+2)/2 + y * (N + 2) + z* (N + 2) * (N + 2), n) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<used_FOUT; q++)
    b_bias(q) = ((float)(rand()%256 - 128)) / 127.f;

  std::cout << "Buffers Initialized" << std::endl;
  for (int i = 0; i < NB_TESTS; i++)
  {
    start = rtclock();
		stride2_spconv(
      b_SIZES.raw_buffer(),
      b_input.raw_buffer(),
      b_filter_values.raw_buffer(),
      b_filter_idx.raw_buffer(),
      b_filter_finptr.raw_buffer(),
      b_bias.raw_buffer(),
      b_result.raw_buffer()
    );
    end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  if (SHOW_OUTPUT)
    print_buffer(b_result);

  print_time("performance_CPU.csv", "spconv",
             {"Tiramisu"},
             {median(duration_vector)});

  if (WRITE_RESULT_TO_FILE){
    // Write results to file
    FILE* f = fopen("tiramisu_result.txt", "w");
    if (f == NULL) {
      printf("Error creating tiramisu_result.txt.\n");
      return 0;
    }

    for(int b=0; b<BATCH_SIZE; b++)
      for(int fout=0; fout<used_FOUT; fout++)
        for(int y=0; y<N/2; y++)
          for(int x=0; x<N/2; x++)
            fprintf(f, "%.17g\n", b_result(x, y, fout, b));

    fclose(f);
  }

  if (CHECK_CORRECTNESS){
    // Compare results with Intel MKL
    std::ifstream mkldnn_result("mkl_result.txt");
    double tmp;
    long nb_correct = 0;

    for(int b=0; b<BATCH_SIZE; b++)
      for(int fout=0; fout<used_FOUT; fout++)
        for(int y=0; y<N/2; y++)
          for(int x=0; x< N/2; x++){
            mkldnn_result >> tmp;
            if (std::abs(b_result(x, y, fout, b) - tmp) <= 0.00001)
              nb_correct++;
          }

    std::cout << "\n\t\tPercentage of correctness " << 100*(((double)nb_correct)/(BATCH_SIZE * used_FOUT * N/2 * N/2)) << "%" << std::endl << std::endl;
  }

  free(filter_idx);
  free(filter_values);
  free(filter_finptr);
  return 0;
}
