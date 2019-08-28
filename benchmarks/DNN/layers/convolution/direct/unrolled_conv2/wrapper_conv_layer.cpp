#include "Halide.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "configure.h"
#include "generated_conv_layer.o.h"
#include <tiramisu/utils.h>

// Original version by: Kyle Spafford Adapted for COO Format
int initRandomSparseMatrix(float* matrix, float density, const int KK, int seed)
{
    const int n = KK * KK * FIn * FOut * density; // number of non zero elements
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * FIn * FOut;
    double prob = (double)n / ((double) total_num_entries);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;
    srand(seed);
    for (int fout_b = 0; fout_b < FOUT_NB_BLOCKS; fout_b++)
    {
      for (int fin_b = 0; fin_b < FIN_NB_BLOCKS; fin_b++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
						for (int ffin = 0; ffin < FIN_BLOCKING; ffin++)
            {
  						for(int ffout = 0; ffout < FOUT_BLOCKING; ffout++)
  						{

								int numEntriesLeft = total_num_entries - ((fout_b * KK * KK * FIn * FOUT_BLOCKING) + (fin_b * KK * KK * FOUT_BLOCKING * FIN_BLOCKING) + (ky * KK * FOUT_BLOCKING * FIN_BLOCKING) + kx * FOUT_BLOCKING * FIN_BLOCKING + ffin * FOUT_BLOCKING + ffout);
								int needToAssign   = n - nnzAssigned;
								if (numEntriesLeft <= needToAssign) {
									fillRemaining = 1;
								}
								if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
								{
									matrix[ffout + ffin * FOUT_BLOCKING + kx * FIN_BLOCKING * FOUT_BLOCKING + ky * FIN_BLOCKING * FOUT_BLOCKING * KK + fin_b * FIN_BLOCKING * FOUT_BLOCKING * KK * KK + fout_b * FOUT_BLOCKING * FIn * KK * KK] = ((float)(rand()%256 - 128)) / 127.f;
									nnzAssigned++;
								}
								else{
									matrix[ffout + ffin * FOUT_BLOCKING + kx * FIN_BLOCKING * FOUT_BLOCKING + ky * FIN_BLOCKING * FOUT_BLOCKING * KK + fin_b * FIN_BLOCKING * FOUT_BLOCKING * KK * KK + fout_b * FOUT_BLOCKING * FIn * KK * KK] = 0;
								}
							}
						}
          }
        }
      }
    }
    if (nnzAssigned != n){
      printf("Error initializing the matrix\n");
      exit(500);
    }

    return n;
}

// The file must be in the CSR-like format (colidx points to input matrix not weights)
void importCSRFromFileAsDense(std::string filename, float** formattedMatrix, int* FOUT, int* FIN, int* KK, int* n){
    std::ifstream cFile (filename);
    float* values;
    int* rowptr;
    int* colidx;
    int NNZ;
		float* matrix;
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
        matrix = (float*)malloc(((*FOUT) * (*FIN) * (*KK) * (*KK)) * sizeof(float));
        memset(matrix, 0.f, ((*FOUT) * (*FIN) * (*KK) * (*KK)) * sizeof(float));
        for (int fout = 0; fout < *FOUT; fout++){
          int fstart = rowptr[fout];
          int fend = rowptr[fout + 1];
          for(int i = fstart; i < fend; i++){
            int fin = colidx[i] / (*n + 2) / (*n + 2);
            int ky = colidx[i] / (*n + 2) % (*n + 2);
            int kx = colidx[i] % (*n + 2);

            matrix[fout * (*FIN) * (*KK) * (*KK) + fin * (*KK) * (*KK) + ky * (*KK) + kx] = values[i];
          }
        }
        free(values);
        free(rowptr);
        free(colidx);
    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

		// Switch to FOUT_NB_BLOCKS, FIN_NB_BLOCKS, K, K, FOUT_BLOCKING, FIN_BLOCKING format
		*formattedMatrix = (float*)malloc(((*FOUT) * (*FIN) * (*KK) * (*KK)) * sizeof(float));
		for (int fout = 0; fout < FOut; fout++)
			for (int fin = 0; fin < FIn; fin++)
				for (int ky = 0; ky < K; ky++)
					for (int kx = 0; kx < K; kx++)
						(*formattedMatrix)[fout%FOUT_BLOCKING + (fin%FIN_BLOCKING) * FOUT_BLOCKING + kx * FIN_BLOCKING * FOUT_BLOCKING + ky * FIN_BLOCKING * FOUT_BLOCKING * (*KK) + (fin/FIN_BLOCKING) * FIN_BLOCKING * FOUT_BLOCKING * (*KK) * (*KK) + (fout/FOUT_BLOCKING) * FOUT_BLOCKING * (*FIN) * (*KK) * (*KK)] = matrix[kx + ky * (*KK) + fin * (*KK) * (*KK) + fout * (*KK) * (*KK) * (*FIN)];
		free(matrix);
}

int main(int, char**)
{
	srand(1);
	std::vector<double> duration_vector;
	double start, end;

	Halide::Buffer<float> input(FIN_BLOCKING, N + 2, N + 2, FIN_NB_BLOCKS, BATCH_SIZE);
	Halide::Buffer<float> bias(FOut);
	Halide::Buffer<float> conv(FOUT_BLOCKING, N, N, FOUT_NB_BLOCKS, BATCH_SIZE);

	// Initialize buffers
	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int fin = 0; fin < FIn; ++fin)
			for (int y = 0; y < N + 2; ++y)
				for (int x = 0; x < N + 2; ++x)
					input(fin%FIN_BLOCKING, x, y, fin/FIN_BLOCKING, n) = ((float)(rand()%256 - 128)) / 127.f;

	int used_FOUT;
	int used_FIN;
	int used_K;
	int n;
	float* weights_buf;
	if (IMPORT_CSR_FROM_FILE){
		importCSRFromFileAsDense("resnet_10.csr", &weights_buf, &used_FOUT, &used_FIN, &used_K, &n);
	}
	else{
		used_FOUT = FOut;
		used_FIN = FIn;
		used_K = K;
		n = N;
		weights_buf = (float*) malloc(FOut * FIn * K * K * sizeof(float));
		initRandomSparseMatrix(weights_buf, WEIGHTS_DENSITY, K, 2);
	}

	// Assertions to ensure that the generated tiramisu code has the right parameters
	// because we are defining the parameters in the configure.h files to get specialized fast code
	assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
	assert((used_FIN == FIn) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
	assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
	assert((n == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

	Halide::Buffer<float> filter(weights_buf, FOUT_BLOCKING, FIN_BLOCKING, K, K, FIN_NB_BLOCKS, FOUT_NB_BLOCKS);

	srand(3);
	for (int fout = 0; fout < FOut; ++fout)
		bias(fout) = ((float)(rand()%256 - 128)) / 127.f;

	std::cout << "\t\tBuffers initialized" << std::endl;

	// Execute Tiramisu code
	for (int i = 0; i < NB_TESTS; i++) {
		start = rtclock();

		conv_tiramisu(
			input.raw_buffer(),
			filter.raw_buffer(),
			bias.raw_buffer(),
			conv.raw_buffer()
		);

		end = rtclock();
		duration_vector.push_back((end - start) * 1000);
	}

	std::cout << "\t\tN = " << N << "; BATCH_SIZE = " << BATCH_SIZE << "; FIn = " << FIn << "; FOut = " << FOut << ";" << std::endl;
	std::cout << "\t\tTiramisu conv" << ": " << median(duration_vector) << "; " << std::endl;

	// Write results to file
	FILE* f = fopen("tiramisu_result.txt", "w");
	if (f == NULL) {
		printf("Error creating tiramisu_result.txt.\n");
		return 0;
	}

	for (int n = 0; n < BATCH_SIZE; ++n)
		for (int fout = 0; fout < FOut; ++fout)
			for (int y = 0; y < N; ++y)
				for (int x = 0; x < N; ++x)
					fprintf(f, "%.10g\n", conv(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n));

	fclose(f);

    // Compare results with Intel MKL
    std::ifstream mkl_result("mkl_result.txt");
    float tmp;
    float file_count = 0, corr = 0;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    mkl_result >> tmp;

                    file_count++;
                    if (std::abs(conv(fout%FOUT_BLOCKING, x, y, fout/FOUT_BLOCKING, n) - tmp) <= 0.001)
                        corr++;
                }

    std::cout << "\t\tResult"
              << ":\n\n";

    std::cout << "\t\tPercentage of correctness " << corr / file_count * 100 << "%" << std::endl << std::endl;

    return 0;
}
