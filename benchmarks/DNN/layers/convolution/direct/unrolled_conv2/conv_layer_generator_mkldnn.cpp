#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include <string>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "mkldnn.hpp"
#include "configure.h"

using namespace mkldnn;
using namespace std;

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
									matrix[kx + ky * KK + (fin_b * FIN_BLOCKING + ffin) * KK * KK + (fout_b * FOUT_BLOCKING + ffout) * KK * KK * FIn] = ((float)(rand()%256 - 128)) / 127.f;
									nnzAssigned++;
								}
								else{
									matrix[kx + ky * KK + (fin_b * FIN_BLOCKING + ffin) * KK * KK + (fout_b * FOUT_BLOCKING + ffout) * KK * KK * FIn] = 0;
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

void conv()
{
    srand(1);
    std::vector<double> duration_vector;
    double start, end;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {0, 0};

    std::vector<float> input_buf(BATCH_SIZE*FIn*(N + 2)*(N + 2));

    std::vector<float> conv_bias_buf(FOut);

    for (int i = 0; i < BATCH_SIZE*FIn*(N + 2)*(N + 2); i++)
        input_buf[i] = ((float)(rand()%256 - 128)) / 127.f;

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

    // create weights array from the pointer
    std::vector<float> conv_weights_buf(weights_buf, weights_buf + FOut * FIn * K * K);

    srand(3);
    for (int i = 0; i < FOut; i++)
        conv_bias_buf[i] = ((float)(rand()%256 - 128)) / 127.f;

    // Create memory objects with user data format
    auto input_usr_md = memory::desc(
        {BATCH_SIZE, FIn, N + 2, N + 2},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto conv_weights_usr_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv_weights_buf.data());
    auto conv_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv_src_md = memory::desc(
        {BATCH_SIZE, FIn, N + 2, N + 2},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_weights_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_bias_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto output_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
        memory::data_type::f32,
        memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv_src_md,
        conv_weights_md,
        conv_bias_md,
        output_md,
        conv_strides,
        conv_padding,
        conv_padding
    );

    auto conv_pd = convolution_forward::primitive_desc(
        conv_d,
        cpu_engine
    );

    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto input_usr_mem = memory(input_usr_md, cpu_engine, input_buf.data());
    auto input_mem = memory(conv_pd.src_desc(), cpu_engine);

    reorder(input_usr_mem, input_mem)
        .execute(cpu_stream, input_usr_mem, input_mem);

    auto conv_weights_mem = conv_weights_usr_mem;
    if (conv_pd.weights_desc() != conv_weights_usr_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), cpu_engine);
        reorder(conv_weights_usr_mem, conv_weights_mem)
            .execute(cpu_stream, conv_weights_usr_mem, conv_weights_mem);
    }

    // Add convolution to the network
    net.push_back(convolution_forward(conv_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
        {MKLDNN_ARG_BIAS, conv_bias_usr_mem},
        {MKLDNN_ARG_DST, conv_dst_mem}
    });

    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        start = rtclock();

        for (size_t j = 0; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);

        cpu_stream.wait();

        end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\n\n\tConv time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, FOut, N, N},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto output_mem = memory(output_usr_md, cpu_engine);
    reorder(conv_dst_mem, output_mem)
        .execute(cpu_stream, conv_dst_mem, output_mem);

    /* Write results to file */
    float* output = (float*)output_mem.get_data_handle();
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        std::cout << "Error creating mkl_result.txt" << std::endl;;
        return ;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output[x + y*N + fout*N*N + n*N*N*FOut]);

    fclose(f);
}

int main(int argc, char **argv)
{
    try {
		conv();
    }

    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
