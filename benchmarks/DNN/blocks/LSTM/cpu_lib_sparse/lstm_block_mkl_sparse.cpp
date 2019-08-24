#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <algorithm>
#include <string.h>
#include "mkl.h"

#include <math.h>
#include "configure.h"
#include "mkl_spblas.h"

#define SIGMOID(x)  (1 / (1 + exp(-(x))))
#define TANH(x) ((exp(2*(x)) - 1) / (exp(2*(x)) + 1))

// Original version by: Kyle Spafford Adapted for CSR format
void initRandomWeights(float* filter_values, MKL_INT* filter_idx, MKL_INT* filter_finptr, float density, const int nb_rows, const int nb_cols, const int seed)
{
    int n = nb_rows * nb_cols * density;
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = nb_rows * nb_cols;
    double prob = (double)n / ((double) total_num_entries);

    // Seed random number generator
    srand(seed);

    // Randomly decide whether entry r,c gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;

    for (int r = 0; r < nb_rows; r++)
    {
      filter_finptr[r] = (MKL_INT)nnzAssigned;
      for (int c = 0; c < nb_cols; c++)
      {
        int numEntriesLeft = total_num_entries - (r * nb_cols + c);
        int needToAssign   = n - nnzAssigned;
        if (numEntriesLeft <= needToAssign) {
            fillRemaining = 1;
        }
        if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
        {
            filter_idx[nnzAssigned] = (MKL_INT)c;
            filter_values[nnzAssigned] =  ((float)(rand()%256 - 128)) / 1270.f;
            nnzAssigned++;
        }
      }
    }
    filter_finptr[nb_rows] = nnzAssigned;
    if (nnzAssigned != n)
      exit(500);
}

// Sparse LSTM block using MKL Sparse (spmv)
void splstm_block(
    sparse_matrix_t* buf_weights,
    struct matrix_descr* buf_weights_descr,
    float* buf_biases,
    float* buf_input,
    float* buf_h,
    float* buf_c,
    float* buf_output
){
  float tmp[BATCH_SIZE * 4 * FEATURE_SIZE * SEQ_LENGTH];
  float sig_buf[4 * VEC_LEN];

  for (int s = 0; s < SEQ_LENGTH; s++){
    for (int i = 0; i < FEATURE_SIZE; i++){
      buf_h[(s + 1) * FEATURE_SIZE + i] = buf_input[s * FEATURE_SIZE + i];
    }
  }
  for (int l = 0; l < NUM_LAYERS; l++){
    for (int i = 0; i < FEATURE_SIZE; i++){
      buf_h[i] = 0;
      buf_c[i] = 0;
    }
    for (int s = 0; s < SEQ_LENGTH; s++)
      mkl_sparse_s_mv(
                  SPARSE_OPERATION_NON_TRANSPOSE,
                  1,
                  buf_weights[l * 2],
                  buf_weights_descr[l * 2],
                  buf_h + (s + 1) * FEATURE_SIZE,
                  0,
                  tmp + s * 4 * FEATURE_SIZE
      );

    for (int s = 0; s < SEQ_LENGTH; s++){
      mkl_sparse_s_mv(
                  SPARSE_OPERATION_NON_TRANSPOSE,
                  1,
                  buf_weights[l * 2 + 1],
                  buf_weights_descr[l * 2 + 1],
                  buf_h + s * FEATURE_SIZE,
                  1,
                  tmp + s * 4 * FEATURE_SIZE
      );
      for (int i = 0; i < FEATURE_SIZE; i++){
        sig_buf[i % VEC_LEN] = SIGMOID(tmp[s * BATCH_SIZE * 4 * FEATURE_SIZE + i] + buf_biases[l * 4 * FEATURE_SIZE + i]);
        sig_buf[VEC_LEN + i % VEC_LEN] = SIGMOID(tmp[s * BATCH_SIZE * 4 * FEATURE_SIZE + i + FEATURE_SIZE] + buf_biases[l * 4 * FEATURE_SIZE + i + FEATURE_SIZE]);
        sig_buf[2 * VEC_LEN + i % VEC_LEN] = TANH(tmp[s * BATCH_SIZE * 4 * FEATURE_SIZE + i + 2 * FEATURE_SIZE] + buf_biases[l * 4 * FEATURE_SIZE + i + 2 * FEATURE_SIZE]);
        sig_buf[3 * VEC_LEN + i % VEC_LEN] = SIGMOID(tmp[s * BATCH_SIZE * 4 * FEATURE_SIZE + i + 3 * FEATURE_SIZE] + buf_biases[l * 4 * FEATURE_SIZE + i + 3 * FEATURE_SIZE]);

        buf_c[i] = sig_buf[i % VEC_LEN] * sig_buf[2 * VEC_LEN + i % VEC_LEN] + sig_buf[VEC_LEN + i % VEC_LEN] * buf_c[i];
        buf_h[(s + 1) * FEATURE_SIZE + i] = TANH(buf_c[i] * sig_buf[3 * VEC_LEN + i % VEC_LEN]);

      }
    }
  }
  for (int s = 0; s < SEQ_LENGTH; s++){
    for (int i = 0; i < FEATURE_SIZE; i++){
      buf_output[s * FEATURE_SIZE + i] = buf_h[(s + 1) * FEATURE_SIZE + i];
    }
  }
}

int main()
{
  int warmupN = 10;

  float* buf_biases = (float*)malloc(4 * FEATURE_SIZE * NUM_LAYERS * sizeof(float));

  float* buf_input = (float*)malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(float));
  float* buf_output = (float*)malloc(FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * sizeof(float));

  float* buf_h = (float*)malloc(FEATURE_SIZE * BATCH_SIZE * (SEQ_LENGTH + 1) * sizeof(float));
  float* buf_c = (float*)malloc(FEATURE_SIZE * BATCH_SIZE * sizeof(float));

  sparse_matrix_t* buf_weights = (sparse_matrix_t*)malloc(2 * NUM_LAYERS * sizeof(sparse_matrix_t));
  struct matrix_descr* buf_weights_descr = (struct matrix_descr*)malloc(2 * NUM_LAYERS * sizeof(struct matrix_descr));

  for (int i =0; i < NUM_LAYERS * 2; i++){
    int FNNZ = 4 * FEATURE_SIZE * FEATURE_SIZE * WEIGHTS_DENSITY;

    float* filter_values = (float*)malloc(FNNZ * sizeof(float));
    MKL_INT* filter_idx = (MKL_INT*)malloc(FNNZ * sizeof(MKL_INT)); //MKL_INT
    MKL_INT* filter_finptr = (MKL_INT*)malloc((4 * FEATURE_SIZE + 1) * sizeof(MKL_INT));;
    // Generate sparse weights matrix
    initRandomWeights(filter_values, filter_idx, filter_finptr, WEIGHTS_DENSITY, 4 * FEATURE_SIZE, FEATURE_SIZE, i);

    // Create handle with matrix stored in CSR format
    mkl_sparse_s_create_csr(&buf_weights[i], SPARSE_INDEX_BASE_ZERO,
                                    4 * FEATURE_SIZE,  // number of rows
                                    FEATURE_SIZE,  // number of cols
                                    filter_finptr,
                                    filter_finptr+1,
                                    filter_idx,
                                    filter_values);

    // Analyze sparse matrix; choose proper kernels and workload balancing strategy
    mkl_sparse_optimize(buf_weights[i]);

    // Set matrix type
    buf_weights_descr[i].type = SPARSE_MATRIX_TYPE_GENERAL;
  }

  // Initialize biases and input
  srand(2);

  for (int i = 0; i < NUM_LAYERS; i++)
      for (int j = 0; j < 4 * FEATURE_SIZE; j++)
          buf_biases[i * 4 * FEATURE_SIZE + j] = ((float)(rand()%256 - 128)) / 1270.f;

  for (int i = 0; i < SEQ_LENGTH; i++)
      for (int j = 0; j < BATCH_SIZE; j++)
          for (int k = 0; k < FEATURE_SIZE; k++)
              buf_input[i * BATCH_SIZE * FEATURE_SIZE + j * FEATURE_SIZE + k] = ((float)(rand()%256 - 128)) / 1270.f;

  std::cout << "Initalization done" << std::endl;

  // Warmup
  for (int i = 0; i < warmupN; i++) {
      splstm_block(
          buf_weights,
          buf_weights_descr,
          buf_biases,
          buf_input,
          buf_h,
          buf_c,
          buf_output
      );
  }

  std::cout << "Warmup done" << std::endl;

  // Execute Tiramisu code
  std::vector<double> durations;
  for (int i = 0; i < NB_TESTS; i++) {
      double start = rtclock();

      splstm_block(
          buf_weights,
          buf_weights_descr,
          buf_biases,
          buf_input,
          buf_h,
          buf_c,
          buf_output
      );

      double end = rtclock();
      durations.push_back((end - start) * 1000);
  }

  std::cout << "MKL LSTM Sparse median runtime: " << median(durations) << " ms" << std::endl << std::flush;

  std::cout << "MKL LSTM Sparse done" << std::endl;

  // Write results to file
  std::ofstream resultfile;
  resultfile.open("mkl_result.txt");

  for (int n = 0; n < SEQ_LENGTH; ++n)
      for (int z = 0; z < BATCH_SIZE; ++z)
          for (int y = 0; y < FEATURE_SIZE; ++y)
              resultfile << std::setprecision(10) << buf_output[n * BATCH_SIZE * FEATURE_SIZE + z * FEATURE_SIZE + y] << std::endl;

  resultfile.close();

  return 0;
}
