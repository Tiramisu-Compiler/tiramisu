#define __TIRAMISU_WRAPPER__
#include "generated_sparse_vgg_block_512_512_28_tiramisu.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "configure.h"

// Original version by: Kyle Spafford Adapted for CSR format
void initRandomWeights(float * filter_values, int* filter_idx, int* filter_finptr, const int n, const int KK, const int fin_size, const int fout_size, int fin_blocking, const int seed)
{
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * fin_size * fout_size;
    double prob = (double)n / ((double) total_num_entries);

    // Seed random number generator
    srand(seed);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;

    for (int fout = 0; fout < fout_size; fout++)
    {
      filter_finptr[fout] = nnzAssigned;
      for (int fin_b = 0; fin_b < fin_size/fin_blocking; fin_b++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
            for (int ffin = 0; ffin < fin_blocking; ffin++)
            {
              int numEntriesLeft = total_num_entries - ((fout * KK * KK * fin_size) + (fin_b * KK * KK * fin_blocking) + (ky * KK * fin_blocking) + kx * fin_blocking + ffin);
              int needToAssign   = n - nnzAssigned;
              if (numEntriesLeft <= needToAssign) {
                  fillRemaining = 1;
              }
              if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
              {
                  filter_idx[nnzAssigned] = (fin_b * fin_blocking + ffin) * (N + 2) * (N + 2) + ky * (N + 2) + kx;
                  filter_values[nnzAssigned] = ((float)(rand()%256 - 128)) / 127.f;
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

int generateCSRWeights(float **filter_values, float density, int **filter_idx, int** filter_finptr, int KK, int fin_size, int fout_size, int fin_blocking, int seed) {
    int nNonzero = KK * KK * fin_size * fout_size * density;
    *filter_values = (float *) malloc(nNonzero * sizeof(float));
    *filter_idx = (int *) malloc(nNonzero * sizeof(int));
    *filter_finptr = (int *) malloc((fout_size + 1) * sizeof(int));
    initRandomWeights(*filter_values, *filter_idx, *filter_finptr, nNonzero, KK, fin_size, fout_size, fin_blocking, seed);
    return nNonzero;
}

int main(int, char **)
{
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------

  // First convolution
  float *filter_values1;
  int *filter_idx1;
  int *filter_finptr1;

  int FNNZ1 = generateCSRWeights(&filter_values1, WEIGHTS_DENSITY, &filter_idx1, &filter_finptr1, K, FIn, FOut, FIN1_BLOCKING, 1);

  Halide::Buffer<float> b_input((N+2) * (N+2) * FIn, BATCH_SIZE);

  Halide::Buffer<float> b_filter_values1(filter_values1, FNNZ1);
  Halide::Buffer<int> b_filter_idx1(filter_idx1, FNNZ1);
  Halide::Buffer<int> b_filter_finptr1(filter_finptr1, FOut + 1);

  Halide::Buffer<float> b_bias1(FOut);

  Halide::Buffer<float> b_conv1(N + 2, N + 2, FOut, BATCH_SIZE);

  // Second convolution
  float *filter_values2;
  int *filter_idx2;
  int *filter_finptr2;

  int FNNZ2 = generateCSRWeights(&filter_values2, WEIGHTS_DENSITY2, &filter_idx2, &filter_finptr2, K, FOut, FOut, FIN2_BLOCKING, 2);

  Halide::Buffer<float> b_filter_values2(filter_values2, FNNZ2);
  Halide::Buffer<int> b_filter_idx2(filter_idx2, FNNZ2);
  Halide::Buffer<int> b_filter_finptr2(filter_finptr2, FOut + 1);

  Halide::Buffer<float> b_bias2(FOut);

  Halide::Buffer<float> b_result(N/2 + 2 * PAD_OUTPUT, N/2 + 2 * PAD_OUTPUT, FOut, BATCH_SIZE);


  Halide::Buffer<int> b_SIZES(2);
  b_SIZES(0) = FNNZ1;
  b_SIZES(1) = FNNZ2;

  srand(3);
  for (int n=0; n < BATCH_SIZE; ++n)
    for (int z=0; z < FIn; ++z)
      for (int y=0; y < N+2; ++y)
        for (int x=0; x < N+2; ++x)
            b_input(x + y * (N+2) + z*(N+2)*(N+2), n) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<FOut; q++)
    b_bias1(q) = ((float)(rand()%256 - 128)) / 127.f;

  for (int q=0; q<FOut; q++)
    b_bias2(q) = ((float)(rand()%256 - 128)) / 127.f;

  for (int i = 0; i < NB_TESTS; i++)
  {
    auto start2 = std::chrono::high_resolution_clock::now();

		sparse_vgg_block_512_512_28_tiramisu(
      b_SIZES.raw_buffer(),
      b_input.raw_buffer(),
      b_filter_values1.raw_buffer(),
      b_filter_idx1.raw_buffer(),
      b_filter_finptr1.raw_buffer(),
      b_bias1.raw_buffer(),
      b_conv1.raw_buffer(),
      b_filter_values2.raw_buffer(),
      b_filter_idx2.raw_buffer(),
      b_filter_finptr2.raw_buffer(),
      b_bias2.raw_buffer(),
      b_result.raw_buffer()
    );

    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    duration_vector_2.push_back(duration2);
  }

  std::cout << ","<< median(duration_vector_2) << ",";

  // Compare results with Intel MKL
  std::ifstream mkldnn_result("mkl_result.txt");
  double tmp;
  long nb_correct = 0;

  for(int b=0; b<BATCH_SIZE; b++)
    for(int fout=0; fout<FOut; fout++)
      for(int y=0; y<N/2; y++)
        for(int x=0; x< N/2; x++){
          mkldnn_result >> tmp;
          if (std::abs(b_result(x + PAD_OUTPUT, y + PAD_OUTPUT, fout, b) - tmp) <= 0.0001)
            nb_correct++;
        }

  if (100*(((double)nb_correct)/(BATCH_SIZE * FOut * (N/2) * (N/2)))<100)
    std::cout << 0;
  else
    std::cout << 1;

  free(filter_idx1);
  free(filter_values1);
  free(filter_finptr1);
  free(filter_idx2);
  free(filter_values2);
  free(filter_finptr2);
  return 0;
}
