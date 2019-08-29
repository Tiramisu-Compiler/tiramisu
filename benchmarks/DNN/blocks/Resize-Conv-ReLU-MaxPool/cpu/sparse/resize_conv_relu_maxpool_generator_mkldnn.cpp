#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "mkldnn.hpp"
#include "mkldnn_debug.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
            if ((nnzAssigned < n && ((double) rand() / (RAND_MAX + 1.0)) <= prob) || fillRemaining)
            {
              matrix[kx + ky*KK + fin*KK*KK + fout*KK*KK*fin_size] = ((float)(rand()%256 - 128)) / 127.f;
              nnzAssigned++;
            }
            else{
              matrix[kx + ky*KK + fin*KK*KK + fout*KK*KK*fin_size] = 0;
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

void resize_conv_block()
{
    srand(1);
    std::vector<double> duration_vector;

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Initialize user buffers
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {0, 0};

    memory::dims pool_strides = {2, 2};
    memory::dims pool_kernel = {2, 2};
    memory::dims pool_padding = {0, 0};

    std::vector<float> input_buf(BATCH_SIZE*FIn*IMG_H*IMG_W);
    std::vector<float> conv_weights_buf(FOut*FIn*K*K);
    std::vector<float> conv_bias_buf(FOut);

    initRandomSparseMatrix(conv_weights_buf.data(), WEIGHTS_DENSITY, K, FIn, FOut, 1);

    srand(2);
    for(int b = 0; b<BATCH_SIZE; b++)
      for (int fin = 0; fin<FIn; fin++)
        for(int y = 0; y<IMG_H; y++)
          for (int x = 0; x < IMG_W; x++)
            input_buf[fin + x*FIn + y*IMG_W*FIn + b*IMG_W*IMG_H*FIn] = ((float)(rand() % 256)) / 255.f;

    for (int i = 0; i < FOut; i++)
      conv_bias_buf[i] = ((float)(rand()%256 - 128)) / 127.f;

    // Create memory objects with user data format
    auto resized_usr_md = memory::desc(
      {BATCH_SIZE, FIn, N + 2, N + 2},
      memory::data_type::f32,
      memory::format_tag::nhwc
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

    auto conv_output_md = memory::desc(
      {BATCH_SIZE, FOut, N, N},
      memory::data_type::f32,
      memory::format_tag::any
    );

    // Create the convolution primitive descriptor, so as to get
    // the data format selected by the primitive.
    auto conv_d = convolution_forward::desc(
      prop_kind::forward_inference,
      algorithm::convolution_direct,
      resized_usr_md,
      conv_weights_md,
      conv_bias_md,
      conv_output_md,
      conv_strides,
      conv_padding,
      conv_padding
    );

    post_ops conv_post_ops;
    conv_post_ops.append_eltwise(1, algorithm::eltwise_relu, 0, 0);

    primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_post_ops);

    auto conv_pd = convolution_forward::primitive_desc(
      conv_d,
      conv_attr,
      cpu_engine
    );

    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);

    // Edit user data format
    auto conv_weights_mem = conv_weights_usr_mem;
    if (conv_pd.weights_desc() != conv_weights_usr_mem.get_desc()) {
      conv_weights_mem = memory(conv_pd.weights_desc(), cpu_engine);
      reorder(conv_weights_usr_mem, conv_weights_mem)
        .execute(cpu_stream, conv_weights_usr_mem, conv_weights_mem);
    }

    // Add convolution to the network
    auto resized_usr_mem = memory(resized_usr_md, cpu_engine);

    net.push_back(convolution_forward(conv_pd));
    net_args.push_back({
      {MKLDNN_ARG_SRC, resized_usr_mem},
      {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
      {MKLDNN_ARG_BIAS, conv_bias_usr_mem},
      {MKLDNN_ARG_DST, conv_dst_mem}
    });

    // Create maxpooling primitive
  auto pool_output_md = memory::desc(
    {BATCH_SIZE, FOut, N/2, N/2},
    memory::data_type::f32,
    memory::format_tag::any
  );

  auto pool_d = pooling_forward::desc(
    prop_kind::forward_inference,
    algorithm::pooling_max,
    conv_pd.dst_desc(),
    pool_output_md,
    pool_strides,
    pool_kernel,
    pool_padding,
    pool_padding
  );

  auto pool_pd = pooling_forward::primitive_desc(
    pool_d,
    cpu_engine
  );

  auto pool_dst_mem = memory(pool_pd.dst_desc(), cpu_engine);

  net.push_back(pooling_forward(pool_pd));
  net_args.push_back({
    {MKLDNN_ARG_SRC, conv_dst_mem},
    {MKLDNN_ARG_DST, pool_dst_mem}
  });

  // Execute the network
  for (int i = 0; i < NB_TESTS; ++i) {
    double start = rtclock();

    // Loop through batch dimension to process each image with OpenCV
    for (int j = 0; j < BATCH_SIZE; ++j) {
      float* input_buf_raw = input_buf.data();
      float* resized_buf_raw = (float*)resized_usr_mem.get_data_handle();

      cv::Mat input_mat(IMG_H, IMG_W, CV_32FC3, (uchar*)&input_buf_raw[j * FIn * IMG_W * IMG_H]);
      cv::Mat resized_mat(N + 2, N + 2, CV_32FC3, (uchar*)&resized_buf_raw[j * FIn * (N + 2) * (N + 2)]);

      cv::resize(input_mat, resized_mat, {N + 2, N + 2}, 0, 0, cv::INTER_LINEAR);
    }

    for (size_t j = 0; j < net.size(); ++j)
      net[j].execute(cpu_stream, net_args[j]);

    cpu_stream.wait();

    double end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  std::cout << "\n\n\tResize-Conv block time : " << median(duration_vector) << " ms." << std::endl;

  // Convert convolution output to user data format
  auto output_usr_md = memory::desc(
    {BATCH_SIZE, FOut, N/2, N/2},
    memory::data_type::f32,
    memory::format_tag::nchw
  );

  auto output_mem = memory(output_usr_md, cpu_engine);
  reorder(pool_dst_mem, output_mem)
    .execute(cpu_stream, pool_dst_mem, output_mem);
  if (WRITE_RESULTS_TO_FILE){
    /* Write results to file */
    float* output = (float*)output_mem.get_data_handle();
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
      std::cout << "Error creating mkl_result.txt" << std::endl;;
      return ;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int fout = 0; fout < FOut; ++fout)
        for (int y = 0; y < N/2; ++y)
          for (int x = 0; x < N/2; ++x)
            fprintf(f, "%.10g\n", output[x + y*(N/2) + fout*(N/2)*(N/2) + n*(N/2)*(N/2)*FOut]);

    fclose(f);
  }
}

int main()
{
  try {
    resize_conv_block();
  }

  catch (error &e) {
    std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
              << "Error status: " << mkldnn_status2str(e.status) << std::endl;

    return 1;
  }

  return 0;
}
