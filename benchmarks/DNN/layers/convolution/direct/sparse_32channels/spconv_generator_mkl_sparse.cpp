#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <string.h>
#include "mkl.h"
#include "configure.h"
#include "im2col.hpp"
#include "mkl_spblas.h"

// Original version by: Kyle Spafford Adapted for CSR format
void initRandomWeights(float *filter_values, MKL_INT* filter_idx, MKL_INT* filter_finptr, const float density, const int KK, const int fin_size, const int fout_size, const int seed)
{
    int nnzAssigned = 0;
    int n =  KK * KK * fin_size * fout_size * density;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * fin_size * fout_size;
    double prob = (double)n / ((double) total_num_entries);

    // Seed random number generator
    srand(seed);

    // Randomly decide whether entry gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;

    for (int fout = 0; fout < fout_size; fout++)
    {
      filter_finptr[fout] = (MKL_INT)nnzAssigned;
      for (int fin_b = 0; fin_b < fin_size/FIN_BL; fin_b++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
            for(int ffin = 0; ffin < FIN_BL; ffin++){
              int numEntriesLeft = total_num_entries - ((fout * KK * KK * fin_size) + (fin_b * KK * KK * FIN_BL) + (ky * KK * FIN_BL) + kx * FIN_BL + ffin);
              int needToAssign   = n - nnzAssigned;
              if (numEntriesLeft <= needToAssign) {
                  fillRemaining = 1;
              }
              if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
              {
                  filter_idx[nnzAssigned] = (MKL_INT)((fin_b * FIN_BL + ffin) * KK * KK + ky * KK + kx);
                  filter_values[nnzAssigned] = ((float)(rand()%256 - 128)) / 127.f;
                  nnzAssigned++;
              }
            }
          }
        }
      }
    }
    filter_finptr[fout_size] = nnzAssigned;
    if (nnzAssigned != n)
      exit(500);
}

int main()
{
  std::vector<double> duration_vector;

  int FNNZ = FOut*FIn*K*K*WEIGHTS_DENSITY;
  float filter_values[FNNZ];
  MKL_INT filter_idx[FNNZ]; //MKL_INT
  MKL_INT filter_finptr[FOut+1];
  // Generate sparse weights matrix
  initRandomWeights(filter_values, filter_idx, filter_finptr, WEIGHTS_DENSITY, K, FIn, FOut, 3);

  // Descriptor of main sparse matrix properties
  struct matrix_descr descrFilter;
  // // Structure with sparse matrix stored in CSR format
  sparse_matrix_t       csrFilter;
  float alpha = 1.0, beta = 0.0;

  // Create handle with matrix stored in CSR format
  mkl_sparse_s_create_csr (&csrFilter, SPARSE_INDEX_BASE_ZERO,
                                  FOut,  // number of rows
                                  FIn*K*K,  // number of cols
                                  filter_finptr,
                                  filter_finptr+1,
                                  filter_idx,
                                  filter_values);

  // Analyze sparse matrix; choose proper kernels and workload balancing strategy
  mkl_sparse_optimize(csrFilter);

  // Create matrix descriptor
  descrFilter.type = SPARSE_MATRIX_TYPE_GENERAL;

  // Allocate buffers
  float* input_buf = (float*)malloc(sizeof(float) * FIn * (N + 2) * (N + 2) * BATCH_SIZE);
  float* conv_bias_buf = (float*)malloc(sizeof(float) * FOut);
  float* result_buf = (float*)malloc(sizeof(float) * FIn * (N) * (N) * K * K * BATCH_SIZE);
  float* output_buf = (float*)malloc(sizeof(float) * FOut * (N) * (N) * BATCH_SIZE);

  srand(2);
  for(int b = 0; b < BATCH_SIZE; ++b)
    for (int fin = 0; fin < FIn; ++fin)
      for (int y = 0; y < N + 2; ++y)
        for (int x = 0; x < N + 2; ++x)
          input_buf[x + y*(N+2) + fin*(N+2)*(N+2) + b*(N+2)*(N+2)*FIn] = ((float)(rand() % 256 - 128)) / 127.f;

  for (int i = 0; i < FOut; i++)
      conv_bias_buf[i] = ((float)(rand()%256 - 128)) / 127.f;

  printf("Buffers Initialized\n");
  omp_set_num_threads(4);
  for (int i = 0; i < NB_TESTS; ++i) {
    double start = rtclock();
    for(int batch = 0; batch<BATCH_SIZE; batch++){
      im2col_cpu(&input_buf[batch*(FIn*(N+2)*(N+2))], FIn,
        N+2, N+2, K, K,
        1, 1,
        &result_buf[batch*(FIn*N*N*K*K)]
      );
      // Filter weights are (FOut) * (FIn * K * K)
      // Input is           (FIn * K * K) * (N * N)
      // The result of the mult is : (FOut) * (N * N)
      // Calculates C = alpha*A*B + beta*C
      mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      alpha,
                      csrFilter,
                      descrFilter,
                      SPARSE_LAYOUT_ROW_MAJOR,
                      &result_buf[batch*(FIn*N*N*K*K)],
                      N*N,
                      N*N,
                      beta,
                      &output_buf[batch*(FOut*N*N)],
                      N*N
      );
      #pragma omp parallel for
      for(int fout = 0; fout<FOut; fout++){
        for(int y=0; y<N; y++)
          for(int x=0; x<N; x++)
            output_buf[batch*(FOut*N*N) + fout*N*N + y*N + x] += conv_bias_buf[fout];
      }
    }

    double end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  std::cout << "\t\tSparse Lowered Convolution time : "
  << median(duration_vector) << " ms" << std::endl;

  if (WRITE_RESULT_TO_FILE){
    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
      printf("Error creating mkl_sparse_result.txt.\n");
      return 0;
    }

    for(int b=0; b<BATCH_SIZE; b++)
      for(int fout=0; fout<FOut; fout++)
        for(int y=0; y<N; y++)
          for(int x=0; x<N; x++)
            fprintf(f, "%.17g\n", output_buf[x + y*N + fout*N*N + b*N*N*FOut]);

    fclose(f);
  }
  mkl_sparse_destroy(csrFilter);
  free(input_buf);
  free(result_buf);
  free(output_buf);
  return 0;
}
