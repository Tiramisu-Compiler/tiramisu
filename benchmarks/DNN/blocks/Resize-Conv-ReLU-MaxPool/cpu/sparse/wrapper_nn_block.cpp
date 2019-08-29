#include <iostream>
#include <cstdlib>
#include <Halide.h>
#include <chrono>
#include <tiramisu/tiramisu.h>

#include "resize_spconv_relu_maxpool_tiramisu.o.h"
#include "configure.h"

using namespace std;

void initRandomWeights(float* filter_values, int* filter_idx, int* filter_finptr, const int n, const int KK, const int fin_size, const int fout_size, const int invert)
{
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * fin_size * fout_size;
    double prob = (double)n / ((double) total_num_entries);

    // Seed random number generator
    srand(1);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;
    for (int fout = 0; fout < fout_size; fout++)
    {
      filter_finptr[fout] = nnzAssigned;
      for (int fin = 0; fin < fin_size; fin++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
            int numEntriesLeft = total_num_entries - ((fout * KK * KK * fin_size) + (fin * KK * KK) + (ky * KK) + kx);
            int needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = 1;
            }
            if ((nnzAssigned < n && ((double) rand() / (RAND_MAX)) <= prob) || fillRemaining)
            {
                filter_idx[nnzAssigned] = fin * (N + 2) * (N + 2) + ky * (N + 2) + kx;
                filter_values[nnzAssigned] = ((float)(rand()%256 - 128)) / 127.f;
                nnzAssigned++;
            }
          }
        }
      }
    }
    filter_finptr[fout_size] = nnzAssigned;
    assert(nnzAssigned == n);
}

int generateCSRWeights(float **filter_values, float density, int **filter_idx, int** filter_finptr, int KK, int fin_size, int fout_size, int invert) {
    int nNonzero = KK * KK * fin_size * fout_size * density;
    *filter_values = (float *) malloc(nNonzero * sizeof(float));
    *filter_idx = (int *) malloc(nNonzero * sizeof(int));
    *filter_finptr = (int *) malloc((fout_size + 1) * sizeof(int));
    initRandomWeights(*filter_values, *filter_idx, *filter_finptr, nNonzero, KK, fin_size, fout_size, invert);
    return nNonzero;
}


int main()
{
    std::vector<double> duration_vector;
    Halide::Buffer<float> input(IMG_W, IMG_H, FIn, BATCH_SIZE);


    float *filter_values;
    int *filter_idx;
    int *filter_finptr;

    int FNNZ = generateCSRWeights(&filter_values, WEIGHTS_DENSITY, &filter_idx, &filter_finptr, K, FIn, FOut, 0);

    Halide::Buffer<int> b_SIZES(1);
    b_SIZES(0) = FNNZ;

    Halide::Buffer<float> resized_input(N + 2, N + 2, FIn, BATCH_SIZE);

    Halide::Buffer<float> b_filter_values(filter_values, FNNZ);
    Halide::Buffer<int> b_filter_idx(filter_idx, FNNZ);
    Halide::Buffer<int> b_filter_finptr(filter_finptr, FOut + 1);

    Halide::Buffer<float> conv_bias(FOut);

    Halide::Buffer<float> output(N/2, N/2, FOut, BATCH_SIZE);

    // Initialize buffers
    srand(2);
    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int fin = 0; fin < FIn; ++fin)
        for (int y = 0; y < IMG_H; ++y)
          for (int x = 0; x < IMG_W; ++x)
            input(x, y, fin, n) = ((float)(rand() % 256)) / 255.f;

    for (int fout = 0; fout < FOut; ++fout)
        conv_bias(fout) = ((float)(rand()%256 - 128)) / 127.f;;

    std::cout << "\t\tBuffers initialized" << std::endl;

    // Execute Tiramisu code
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();
        resize_spconv_relu_maxpool_block(
            input.raw_buffer(),
            b_SIZES.raw_buffer(),
            resized_input.raw_buffer(),
            b_filter_values.raw_buffer(),
            b_filter_idx.raw_buffer(),
            b_filter_finptr.raw_buffer(),
            conv_bias.raw_buffer(),
            output.raw_buffer()
        );

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tTiramisu Resize-Conv block duration"
              << ": " << median(duration_vector) << " ms;" << std::endl;
    if (WRITE_RESULTS_TO_FILE){
      // Write results to file
      FILE* f = fopen("tiramisu_result.txt", "w");
      if (f == NULL) {
          printf("Error creating mkl_result.txt.\n");
          return 0;
      }

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N/2; ++y)
                  for (int x = 0; x < N/2; ++x)
                  fprintf(f, "%.10g\t (%d, %d, %d, %d)\n", output(x, y, fout, n), n, fout, y, x);
                      //fprintf(f, "%.10g\n", output(x, y, fout, n)); // DEBUGMODE fprintf(f, "%.10g\t (%d, %d, %d, %d)\n", output(x, y, fout, n), n, fout, y, x);

      fclose(f);
    }

    if (CHECK_CORRECTNESS){
      // Compare results with Intel MKL
      std::ifstream mkl_result("mkl_result.txt");
      float tmp;
      float file_count = 0, corr = 0;

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N/2; ++y)
                  for (int x = 0; x < N/2; ++x) {
                      mkl_result >> tmp;

                      file_count++;
                      if (std::abs(output(x, y, fout, n) - tmp) <= 0.001)
                          corr++;
                      else
                        printf("%f  and  %f     (%d, %d, %d, %d)\n", output(x%(X_BL/2), y%(Y_BL/2), x/(X_BL/2), y/(Y_BL/2), fout, n), tmp, n, fout, y, x );
                  }

      std::cout << "\t\tResult"
                << ":\n\n";

      std::cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << std::endl << std::endl;
    }


    return 0;
}
