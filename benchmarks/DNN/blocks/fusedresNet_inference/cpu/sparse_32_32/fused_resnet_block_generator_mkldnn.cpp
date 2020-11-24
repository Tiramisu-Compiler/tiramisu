#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <assert.h>
#include <string.h>
#include <fstream>

#include "mkldnn.hpp"
#include "mkldnn_debug.h"

#include "configure.h"

using namespace mkldnn;

// Original version by: Kyle Spafford Adapted for COO Format
int initRandomSparseMatrix(float* matrix, float density, const int KK, const int fin_size, const int fout_size, int seed)
{
    const int n = KK * KK * fin_size * fout_size * density; // number of non zero elements
    int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    int total_num_entries = KK * KK * fin_size * fout_size;
    double prob = (double)n / ((double) total_num_entries);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    int fillRemaining = 0;
    srand(seed);
    for (int fout = 0; fout < fout_size; fout++)
    {
      for (int fin_b = 0; fin_b < fin_size/FIN_BL; fin_b++)
      {
        for (int ky = 0; ky < KK; ky++)
        {
          for (int kx = 0; kx < KK; kx++)
          {
            for (int ffin = 0; ffin < FIN_BL; ffin++){
              int numEntriesLeft = total_num_entries - ((fout * KK * KK * fin_size) + (fin_b * KK * KK * FIN_BL) + (ky * KK * FIN_BL) + kx * FIN_BL + ffin);
              int needToAssign   = n - nnzAssigned;
              if (numEntriesLeft <= needToAssign) {
                fillRemaining = 1;
              }
              if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
              {
                // Assign (kx,ky,fin,b) a value
                matrix[kx + ky*KK + (fin_b * FIN_BL + ffin)*KK*KK + fout*KK*KK*fin_size] = ((float)(rand()%256 - 128)) / 127.f;
                nnzAssigned++;
              }
              else{
                matrix[kx + ky*KK + (fin_b * FIN_BL + ffin)*KK*KK + fout*KK*KK*fin_size] = 0;
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

void resnet_block()
{
    std::string filename1 = "resnet_10.csr"; // Layer 1's weights
    std::string filename2 = "resnet_10.csr"; // Layer 2's weights

    std::vector<double> duration_vector;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv1_strides = {STRIDE, STRIDE};
    memory::dims conv2_strides = {1, 1};
    std::vector<float> bn_scale_shift_buf(2*FOut);
    std::vector<float> bn_mean_buf(FOut);
    std::vector<float> bn_variance_buf(FOut);
    std::vector<float> bn2_scale_shift_buf(2*FOut);
    std::vector<float> bn2_mean_buf(FOut);
    std::vector<float> bn2_variance_buf(FOut);

    std::vector<float> input_buf(BATCH_SIZE*FIn*(N+2)*(N+2));

    std::vector<float> conv1_bias_buf(FOut);
    memory::dims conv1_padding = {0, 0};

    std::vector<float> conv2_bias_buf(FOut);

    memory::dims conv2_padding = {1, 1};
    int used_FOUT;
    int used_FIN;
    int used_K;
    int n;
    float* weights1_buf;
    if (IMPORT_CSR_FROM_FILE){
      importCSRFromFileAsDense(filename1, &weights1_buf, &used_FOUT, &used_FIN, &used_K, &n);
    }
    else{
      used_FOUT = FOut;
      used_FIN = FIn;
      used_K = K;
      n = N;
      weights1_buf = (float*) malloc(FOut * FIn * K * K * sizeof(float));
      initRandomSparseMatrix(weights1_buf, WEIGHTS_DENSITY, K, FIn, FOut, 2);
    }

    // Assertions to ensure that the generated tiramisu code has the right parameters
    // because we are defining the parameters in the configure.h files to get specialized fast code
    assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
    assert((used_FIN == FIn) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
    assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
    assert((n == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

    float* weights2_buf;
    if (IMPORT_CSR_FROM_FILE){
      importCSRFromFileAsDense(filename2, &weights2_buf, &used_FOUT, &used_FIN, &used_K, &n);
    }
    else{
      used_FOUT = FOut;
      used_FIN = FOut;
      used_K = K;
      n = N;
      weights2_buf = (float*) malloc(FOut * FOut * K * K * sizeof(float));
      initRandomSparseMatrix(weights2_buf, WEIGHTS_DENSITY2, K, FOut, FOut, 5);
    }

    // Assertions to ensure that the generated tiramisu code has the right parameters
    // because we are defining the parameters in the configure.h files to get specialized fast code
    assert((used_FOUT == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));
    assert((used_FIN == FOut) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));
    assert((used_K == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));
    assert((n == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));

    std::vector<float> conv1_weights_buf(weights1_buf, weights1_buf + FOut*FIn*K*K);
    std::vector<float> conv2_weights_buf(weights2_buf, weights2_buf + FOut*FOut*K*K);

    srand(3);
    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int fin = 0; fin < FIn; ++fin)
        for (int y = 0; y < N + 2; ++y)
          for (int x = 0; x < N + 2; ++x)
            input_buf[x + y*(N+2) + fin*(N+2)*(N+2) + n*(N+2)*(N+2)*FIn] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
      conv1_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout) {
      bn_scale_shift_buf[fout] = 1;
      bn_scale_shift_buf[fout + FOut] = 0;
      bn_mean_buf[fout] = ((float)(rand()%256)) / 127.f;
      bn_variance_buf[fout] = ((float)(rand()%256)) / 127.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
      conv2_bias_buf[fout] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout) {
      bn2_scale_shift_buf[fout] = 1;
      bn2_scale_shift_buf[fout + FOut] = 0;
      bn2_mean_buf[fout] = ((float)(rand()%256)) / 127.f;
      bn2_variance_buf[fout] = ((float)(rand()%256)) / 127.f;
    }

    // Create first convolution primitive
    // Create memory objects with user data format
    auto input_usr_md = memory::desc(
        {BATCH_SIZE, FIn, N+2, N+2},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto conv_weights_usr_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv2_weights_usr_md = memory::desc(
        {FOut, FOut, K, K},
        memory::data_type::f32,
        memory::format_tag::oihw
    );

    auto conv_bias_usr_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto conv1_weights_usr_mem = memory(conv_weights_usr_md, cpu_engine, conv1_weights_buf.data());
    auto conv1_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv1_bias_buf.data());

    // Create memory objects with a data format selected by the convolution primitive
    auto conv1_src_md = memory::desc(
        {BATCH_SIZE, FIn, N+2, N+2},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_weights_md = memory::desc(
        {FOut, FIn, K, K},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv2_weights_md = memory::desc(
        {FOut, FOut, K, K},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_bias_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_output_md = memory::desc(
        {BATCH_SIZE, FOut, N/STRIDE, N/STRIDE},
        memory::data_type::f32,
        memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv1_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv1_src_md,
        conv_weights_md,
        conv_bias_md,
        conv_output_md,
        conv1_strides,
        conv1_padding,
        conv1_padding
    );

    auto conv1_pd = convolution_forward::primitive_desc(
        conv1_d,
        cpu_engine
    );

    auto conv1_dst_mem = memory(conv1_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto input_mem = memory(conv1_pd.src_desc(), cpu_engine);
    auto user_input_mem = memory(input_usr_md, cpu_engine, input_buf.data());

    reorder(user_input_mem, input_mem)
      .execute(cpu_stream, user_input_mem, input_mem);

    auto conv1_weights_mem = conv1_weights_usr_mem;
    if (conv1_pd.weights_desc() != conv1_weights_usr_mem.get_desc()) {
        conv1_weights_mem = memory(conv1_pd.weights_desc(), cpu_engine);
        reorder(conv1_weights_usr_mem, conv1_weights_mem)
          .execute(cpu_stream, conv1_weights_usr_mem, conv1_weights_mem);
    }

    net.push_back(convolution_forward(conv1_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, input_mem},
        {MKLDNN_ARG_WEIGHTS, conv1_weights_mem},
        {MKLDNN_ARG_BIAS, conv1_bias_usr_mem},
        {MKLDNN_ARG_DST, conv1_dst_mem}
    });

    // Create BN fused with ReLU primitive
    auto bn_scale_md = memory::desc(
        {2, FOut},
        memory::data_type::f32,
        memory::format_tag::nc
    );

    auto bn_mean_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto bn_variance_md = memory::desc(
        {FOut},
        memory::data_type::f32,
        memory::format_tag::x
    );

    auto bn1_scale_mem = memory(bn_scale_md, cpu_engine, bn_scale_shift_buf.data());
    auto bn1_mean_mem = memory(bn_mean_md, cpu_engine, bn_mean_buf.data());
    auto bn1_variance_mem = memory(bn_variance_md, cpu_engine, bn_variance_buf.data());

    auto bn1_d = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        conv1_pd.dst_desc(),
        EPSILON,
        mkldnn::normalization_flags::use_scale_shift | mkldnn::normalization_flags::fuse_norm_relu | mkldnn::normalization_flags::use_global_stats
    );

    auto bn1_pd = batch_normalization_forward::primitive_desc(
        bn1_d, cpu_engine
    );

    net.push_back(batch_normalization_forward(bn1_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv1_dst_mem},
        {MKLDNN_ARG_SCALE_SHIFT, bn1_scale_mem},
        {MKLDNN_ARG_MEAN, bn1_mean_mem},
        {MKLDNN_ARG_VARIANCE, bn1_variance_mem},
        {MKLDNN_ARG_DST, conv1_dst_mem}
    });

    // Create second convolution primitive

    // Create memory objects with user data format
    auto conv2_weights_usr_mem = memory(conv2_weights_usr_md, cpu_engine, conv2_weights_buf.data());
    auto conv2_bias_usr_mem = memory(conv_bias_usr_md, cpu_engine, conv2_bias_buf.data());

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv2_d = convolution_forward::desc(
        prop_kind::forward_inference,
        algorithm::convolution_direct,
        conv1_pd.dst_desc(),
        conv2_weights_md,
        conv_bias_md,
        conv_output_md,
        conv2_strides,
        conv2_padding,
        conv2_padding
    );

    auto conv2_pd = convolution_forward::primitive_desc(
        conv2_d,
        cpu_engine
    );

    auto conv2_dst_mem = memory(conv2_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto conv2_weights_mem = conv2_weights_usr_mem;
    if (conv2_pd.weights_desc() != conv2_weights_usr_mem.get_desc()) {
        conv2_weights_mem = memory(conv2_pd.weights_desc(), cpu_engine);
        reorder(conv2_weights_usr_mem, conv2_weights_mem)
            .execute(cpu_stream, conv2_weights_usr_mem, conv2_weights_mem);
    }

    net.push_back(convolution_forward(conv2_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv1_dst_mem},
        {MKLDNN_ARG_WEIGHTS, conv2_weights_mem},
        {MKLDNN_ARG_BIAS, conv2_bias_usr_mem},
        {MKLDNN_ARG_DST, conv2_dst_mem}
    });

    // Create BN fused with ReLU primitive
    auto bn2_scale_mem = memory(bn_scale_md, cpu_engine, bn2_scale_shift_buf.data());
    auto bn2_mean_mem = memory(bn_mean_md, cpu_engine, bn2_mean_buf.data());
    auto bn2_variance_mem = memory(bn_variance_md, cpu_engine, bn2_variance_buf.data());

    auto bn2_d = batch_normalization_forward::desc(
        prop_kind::forward_inference,
        conv2_pd.dst_desc(),
        EPSILON,
        mkldnn::normalization_flags::use_scale_shift | mkldnn::normalization_flags::use_global_stats
    );

    auto bn2_pd = batch_normalization_forward::primitive_desc(
        bn2_d, cpu_engine
    );

    net.push_back(batch_normalization_forward(bn2_pd));
    net_args.push_back({
        {MKLDNN_ARG_SRC, conv2_dst_mem},
        {MKLDNN_ARG_SCALE_SHIFT, bn2_scale_mem},
        {MKLDNN_ARG_MEAN, bn2_mean_mem},
        {MKLDNN_ARG_VARIANCE, bn2_variance_mem},
        {MKLDNN_ARG_DST, conv2_dst_mem}
    });

    // Execute the network
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();

        for (size_t j = 0; j < net.size(); ++j)
            net[j].execute(cpu_stream, net_args[j]);

        cpu_stream.wait();

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\n\n\tResNet block time : " << median(duration_vector) << " ms." << std::endl;

    // Convert convolution output to user data format
    auto output_usr_md = memory::desc(
        {BATCH_SIZE, FOut, N/STRIDE, N/STRIDE},
        memory::data_type::f32,
        memory::format_tag::nchw
    );

    auto output_mem = memory(output_usr_md, cpu_engine);
    reorder(conv2_dst_mem, output_mem)
      .execute(cpu_stream, conv2_dst_mem, output_mem);
    if (CHECK_CORRECTNESS){
      /* Write results to file */
      float* output = (float*)output_mem.get_data_handle();
      FILE* f = fopen("mkl_result.txt", "w");
      if (f == NULL) {
          std::cout << "Error creating mkl_result.txt" << std::endl;;
          return ;
      }

      for (int n = 0; n < BATCH_SIZE; ++n)
          for (int fout = 0; fout < FOut; ++fout)
              for (int y = 0; y < N/STRIDE; ++y)
                  for (int x = 0; x < N/STRIDE; ++x)
                      fprintf(f, "%.17g\n", output[x + y*N/STRIDE + fout*N/STRIDE*N/STRIDE + n*N/STRIDE*N/STRIDE*FOut]);

      fclose(f);
    }
}

int main()
{
    try {
        resnet_block();
    }

    catch (error &e) {
        std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
                  << "Error status: " << mkldnn_status2str(e.status) << std::endl;

        return 1;
    }

    return 0;
}
