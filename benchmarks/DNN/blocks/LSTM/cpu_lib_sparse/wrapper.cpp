/*
 * You need to execute the MKL configuration script before
 * compiling this benchmark.
 * For example, if it is located in /opt/intel/mkl/bin/mklvars.sh and you are in 64-bits :
 * source /opt/intel/mkl/bin/mklvars.sh intel64
 */

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "mkl.h"

#include "configure.h"
#include "mkl_spblas.h"
#include "lstm_sparse.o.h"

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

    // Randomly decide whether entry i,j gets a value, but ensure n values
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

int main(int argc, char *argv[])
{
    int warmupN = 10;
    if (argc > 1)
        warmupN = atoi(argv[1]);

    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<DATA_TYPE> buf_biases(4 * FEATURE_SIZE, NUM_LAYERS);

    Halide::Buffer<DATA_TYPE> buf_input(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);
    Halide::Buffer<DATA_TYPE> buf_output(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH);

    Halide::Buffer<DATA_TYPE> buf_h(FEATURE_SIZE, BATCH_SIZE, SEQ_LENGTH + 1);
    Halide::Buffer<DATA_TYPE> buf_c(FEATURE_SIZE, BATCH_SIZE);

    Halide::Buffer<float*> buf_weights(2, NUM_LAYERS);
    Halide::Buffer<float*> buf_weights_descr(2, NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS * 2; i++){
      int FNNZ = 4 * FEATURE_SIZE * FEATURE_SIZE * WEIGHTS_DENSITY;
      float* filter_values = (float*)malloc(FNNZ * sizeof(float));
      MKL_INT* filter_idx = (MKL_INT*)malloc(FNNZ * sizeof(MKL_INT)); //MKL_INT
      MKL_INT* filter_finptr = (MKL_INT*)malloc((4 * FEATURE_SIZE + 1) * sizeof(MKL_INT));;
      // Generate sparse weights matrix
      initRandomWeights(filter_values, filter_idx, filter_finptr, WEIGHTS_DENSITY, 4 * FEATURE_SIZE, FEATURE_SIZE, i);

      // Descriptor of main sparse matrix properties
      struct matrix_descr* descrFilter = (struct matrix_descr*)malloc(sizeof(struct matrix_descr));
      // // Structure with sparse matrix stored in CSR format
      sparse_matrix_t* csrFilter = (sparse_matrix_t*)malloc(sizeof(sparse_matrix_t));;

      // Create handle with matrix stored in CSR format
      mkl_sparse_s_create_csr(csrFilter, SPARSE_INDEX_BASE_ZERO,
                                      4 * FEATURE_SIZE,  // number of rows
                                      FEATURE_SIZE,  // number of cols
                                      filter_finptr,
                                      filter_finptr+1,
                                      filter_idx,
                                      filter_values);

      // Analyze sparse matrix; choose proper kernels and workload balancing strategy
      mkl_sparse_optimize(*csrFilter);

      // Create matrix descriptor
      descrFilter->type = SPARSE_MATRIX_TYPE_GENERAL;

      buf_weights(i%2, i/2) = (float*)csrFilter;
      buf_weights_descr(i%2, i/2) = (float*)descrFilter;
    }
    // Initialize biases and input
    srand(2);

    for (int i = 0; i < NUM_LAYERS; i++)
        for (int j = 0; j < 4 * FEATURE_SIZE; j++)
            buf_biases(j, i) = ((float)(rand()%256 - 128)) / 1270.f;


    for (int i = 0; i < SEQ_LENGTH; i++)
        for (int j = 0; j < BATCH_SIZE; j++)
            for (int k = 0; k < FEATURE_SIZE; k++)
                buf_input(k, j, i) = ((float)(rand()%256 - 128)) / 1270.f;

    std::cout << "Initalization done" << std::endl;

    // Warmup
    for (int i = 0; i < warmupN; i++) {
        lstm_sparse(
            buf_weights.raw_buffer(),
            buf_weights_descr.raw_buffer(),
            buf_biases.raw_buffer(),
            buf_input.raw_buffer(),
            buf_h.raw_buffer(),
            buf_c.raw_buffer(),
            buf_output.raw_buffer()
        );
    }

    std::cout << "Warmup done" << std::endl;

    // Execute Tiramisu code
    std::vector<double> durations;
    for (int i = 0; i < NB_TESTS; i++) {
        double start = rtclock();

        lstm_sparse(
            buf_weights.raw_buffer(),
            buf_weights_descr.raw_buffer(),
            buf_biases.raw_buffer(),
            buf_input.raw_buffer(),
            buf_h.raw_buffer(),
            buf_c.raw_buffer(),
            buf_output.raw_buffer()
        );

        double end = rtclock();
        durations.push_back((end - start) * 1000);
    }

    std::cout << "LSTM Sparse median runtime: " << median(durations) << " ms" << std::endl << std::flush;

    std::cout << "LSTM Sparse done" << std::endl;

    // Write results to file
    std::ofstream resultfile;
    resultfile.open("tiramisu_result.txt");

    for (int n = 0; n < SEQ_LENGTH; ++n)
        for (int z = 0; z < BATCH_SIZE; ++z)
            for (int y = 0; y < FEATURE_SIZE; ++y)
                resultfile << std::setprecision(10) << buf_output(y, z, n) << std::endl;

    resultfile.close();

    std::cout << "\t\t Result"
              << ":\n\n";

    // Check for correctness with MKLDNN or MKL Sparse
    std::ifstream infile1("tiramisu_result.txt"), infile2("mkl_result.txt");
    std::string line1, line2;
    float file_count = 0, corr = 0, f1, f2;

    while (std::getline(infile1, line1))
    {
        std::getline(infile2, line2);
        file_count += 1;
        f1 = std::stof(line1);
        f2 = std::stof(line2);

        if (std::abs(f1 - f2) < 0.002)
            corr += 1;
    }

    printf("\t\t Percentage of correctness %f \n\n", corr / file_count * 100);

    return 0;
}
