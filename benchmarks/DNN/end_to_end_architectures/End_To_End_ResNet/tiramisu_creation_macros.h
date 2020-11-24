#define create_sparse_resnet_block_tiramisu(block_number,filename,filename2,BS,FOUT,FIN,N,K,stride,input,output) \
float *filter_values##_##block_number;\
int *filter_idx##_##block_number;\
int *filter_finptr##_##block_number;\
\
int FNNZ##_##block_number;\
int used_FOUT##_##block_number;\
int used_FIN##_##block_number;\
int used_K##_##block_number;\
int n##_##block_number;\
importCSRFromFile(filename, &filter_values##_##block_number, &filter_finptr##_##block_number, &filter_idx##_##block_number, &used_FOUT##_##block_number, &used_FIN##_##block_number, &used_K##_##block_number, &FNNZ##_##block_number, &n##_##block_number);\
\
assert((used_FOUT##_##block_number == FOUT) && ("FOUT parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_##block_number == FIN) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
Halide::Buffer<int> b_SIZES##_##block_number(2);\
b_SIZES##_##block_number(0) = FNNZ##_##block_number;\
Halide::Buffer<float> b_input##_##block_number(input,(N+2) * (N+2) * FIN, BS);\
\
Halide::Buffer<float> b_filter_values##_##block_number(filter_values##_##block_number, FNNZ##_##block_number);\
Halide::Buffer<int> b_filter_idx##_##block_number(filter_idx##_##block_number, FNNZ##_##block_number);\
Halide::Buffer<int> b_filter_finptr##_##block_number(filter_finptr##_##block_number, FOUT + 1);\
\
Halide::Buffer<float> b_bias##_##block_number(FOUT);\
Halide::Buffer<float> b_bn_scale##_##block_number(FOUT);\
Halide::Buffer<float> b_bn_shift##_##block_number(FOUT);\
Halide::Buffer<float> b_bn_mean##_##block_number(FOUT);\
Halide::Buffer<float> b_bn_variance##_##block_number(FOUT);\
\
Halide::Buffer<float> b_conv1_result##_##block_number(N/stride + 2, N/stride + 2, FOUT, BS);\
\
float *filter_values2##_##block_number;\
int *filter_idx2##_##block_number;\
int *filter_finptr2##_##block_number;\
\
int FNNZ2##_##block_number;\
importCSRFromFile(filename2, &filter_values2##_##block_number, &filter_finptr2##_##block_number, &filter_idx2##_##block_number, &used_FOUT##_##block_number, &used_FIN##_##block_number, &used_K##_##block_number, &FNNZ2##_##block_number, &n##_##block_number);\
\
assert((used_FOUT##_##block_number == FOUT) && ("FOUT parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_##block_number == FOUT) && ("FOUT parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_##block_number == N / stride) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
b_SIZES##_##block_number(1) = FNNZ2##_##block_number;\
Halide::Buffer<float> b_filter_values2##_##block_number(filter_values2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_idx2##_##block_number(filter_idx2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_finptr2##_##block_number(filter_finptr2##_##block_number, FOUT + 1);\
\
Halide::Buffer<float> b_bias2##_##block_number(FOUT);\
Halide::Buffer<float> b_bn2_scale##_##block_number(FOUT);\
Halide::Buffer<float> b_bn2_shift##_##block_number(FOUT);\
Halide::Buffer<float> b_bn2_mean##_##block_number(FOUT);\
Halide::Buffer<float> b_bn2_variance##_##block_number(FOUT);\
\
Halide::Buffer<float> b_result##_##block_number(output, N / stride + 2, N / stride + 2, FOUT, BS);\
\
srand(2);\
\
for (int q=0; q<FOUT; q++)\
  b_bias##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;\
\
for (int q=0; q<FOUT; q++){\
  b_bn_scale##_##block_number(q) = 1.f;\
  b_bn_shift##_##block_number(q) = 0.f;\
  b_bn_mean##_##block_number(q) = ((float)(rand()%256)) / 127.f;\
  b_bn_variance##_##block_number(q) = ((float)(rand()%256)) / 127.f;\
}\
\
for (int q=0; q<FOUT; q++)\
  b_bias2##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;\
\
for (int q=0; q<FOUT; q++){\
  b_bn2_scale##_##block_number(q) = 1.f;\
  b_bn2_shift##_##block_number(q) = 0.f;\
  b_bn2_mean##_##block_number(q) = ((float)(rand()%256)) / 127.f;\
  b_bn2_variance##_##block_number(q) = ((float)(rand()%256)) / 127.f;\
}

#define call_sparse_resnet_block_tiramisu(block_number,function_extension)\
fused_sparse_resnet_block##function_extension(\
  b_SIZES##_##block_number.raw_buffer(),\
  b_input##_##block_number.raw_buffer(),\
    b_filter_values##_##block_number.raw_buffer(),\
    b_filter_idx##_##block_number.raw_buffer(),\
    b_filter_finptr##_##block_number.raw_buffer(),\
    b_bias##_##block_number.raw_buffer(),\
    b_bn_scale##_##block_number.raw_buffer(),\
    b_bn_shift##_##block_number.raw_buffer(),\
    b_bn_mean##_##block_number.raw_buffer(),\
    b_bn_variance##_##block_number.raw_buffer(),\
  b_conv1_result##_##block_number.raw_buffer(),\
    b_filter_values2##_##block_number.raw_buffer(),\
    b_filter_idx2##_##block_number.raw_buffer(),\
    b_filter_finptr2##_##block_number.raw_buffer(),\
    b_bias2##_##block_number.raw_buffer(),\
    b_bn2_scale##_##block_number.raw_buffer(),\
    b_bn2_shift##_##block_number.raw_buffer(),\
    b_bn2_mean##_##block_number.raw_buffer(),\
    b_bn2_variance##_##block_number.raw_buffer(),\
  b_result##_##block_number.raw_buffer()\
);

#define create_conv_relu_maxpool_3_16(block_number,filename,BS,FOUT,FIN,N,K,input,output) \
float *filter_values##_spconv_relu_maxpool_##block_number;\
int *filter_idx##_spconv_relu_maxpool_##block_number;\
int *filter_finptr##_spconv_relu_maxpool_##block_number;\
\
int FNNZ##_spconv_relu_maxpool_##block_number;\
int used_FOUT##_spconv_relu_maxpool_##block_number;\
int used_FIN##_spconv_relu_maxpool_##block_number;\
int used_K##_spconv_relu_maxpool_##block_number;\
int n##_spconv_relu_maxpool_##block_number;\
importCSRFromFile(filename, &filter_values##_spconv_relu_maxpool_##block_number, &filter_finptr##_spconv_relu_maxpool_##block_number, &filter_idx##_spconv_relu_maxpool_##block_number, &used_FOUT##_spconv_relu_maxpool_##block_number, &used_FIN##_spconv_relu_maxpool_##block_number, &used_K##_spconv_relu_maxpool_##block_number, &FNNZ##_spconv_relu_maxpool_##block_number, &n##_spconv_relu_maxpool_##block_number);\
\
assert((used_FOUT##_spconv_relu_maxpool_##block_number == FOUT) && ("FOUT parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_spconv_relu_maxpool_##block_number == FIN) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_spconv_relu_maxpool_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_spconv_relu_maxpool_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
Halide::Buffer<int> b_SIZES##_spconv_relu_maxpool_##block_number(1);\
b_SIZES##_spconv_relu_maxpool_##block_number(0) = FNNZ##_spconv_relu_maxpool_##block_number;\
Halide::Buffer<float> b_input##_spconv_relu_maxpool_##block_number(input, (N + 2) * (N + 2) * FIN, BS);\
\
Halide::Buffer<float> b_result##_spconv_relu_maxpool_##block_number(output, N/2 + 2, N/2 + 2, FOUT, BS);\
Halide::Buffer<float> b_workspace##_spconv_relu_maxpool_##block_number(N, N, FOUT, BS);\
\
Halide::Buffer<float> b_filter_values##_spconv_relu_maxpool_##block_number(filter_values##_spconv_relu_maxpool_##block_number, FNNZ##_spconv_relu_maxpool_##block_number);\
Halide::Buffer<int> b_filter_idx##_spconv_relu_maxpool_##block_number(filter_idx##_spconv_relu_maxpool_##block_number, FNNZ##_spconv_relu_maxpool_##block_number);\
Halide::Buffer<int> b_filter_finptr##_spconv_relu_maxpool_##block_number(filter_finptr##_spconv_relu_maxpool_##block_number, FOUT + 1);\
\
Halide::Buffer<float> b_bias##_spconv_relu_maxpool_##block_number(FOUT);\
\
srand(2);\
\
for (int q=0; q<FOUT; q++)\
  b_bias##_spconv_relu_maxpool_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;

#define call_spconv_relu_maxpool_tiramisu(block_number)\
spconv_relu_maxpool(b_SIZES##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_input##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_filter_values##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_filter_idx##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_filter_finptr##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_bias##_spconv_relu_maxpool_##block_number.raw_buffer(),\
                    b_result##_spconv_relu_maxpool_##block_number.raw_buffer()\
                  );

#define convolution_mkldnn_tiramisu_format(block_number,import_from_file,filename,density,BS,FOUT,FIN,N,K,stride,padding,already_added_padding,input,output,output_resnet,N_OUT,fuse_relu)\
memory::dims conv_strides##_conv_##block_number = {stride, stride};\
memory::dims conv_padding##_conv_##block_number = {padding, padding};\
\
std::vector<float> conv_bias_buf##_conv_##block_number(FOUT);\
\
int used_FOUT##_conv_##block_number;\
int used_FIN##_conv_##block_number;\
int used_K##_conv_##block_number;\
int n##_conv_##block_number;\
float* weights_buf##_conv_##block_number;\
if (import_from_file){\
  importCSRFromFileAsDense(filename, &weights_buf##_conv_##block_number, &used_FOUT##_conv_##block_number, &used_FIN##_conv_##block_number, &used_K##_conv_##block_number, &n##_conv_##block_number);\
}\
else{\
  used_FOUT##_conv_##block_number = FOUT;\
  used_FIN##_conv_##block_number = FIN;\
  used_K##_conv_##block_number = K;\
  n##_conv_##block_number = N;\
  weights_buf##_conv_##block_number = (float*) malloc(FOUT * FIN * K * K * sizeof(float));\
  srand(2);\
  for (int i = 0; i < FOUT * FIN * K * K; i++)\
      weights_buf##_conv_##block_number[i] = ((float)(rand()%256 - 128)) / 127.f;\
}\
\
assert((used_FOUT##_conv_##block_number == FOUT) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_conv_##block_number == FIN) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_conv_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_conv_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
std::vector<float> conv_weights_buf##_conv_##block_number(weights_buf##_conv_##block_number, weights_buf##_conv_##block_number + FOUT * FIN * K * K);\
\
srand(2);\
\
if (K == 1 && stride == 2)\
  for (int i = 0; i < FOUT; i++)\
      conv_bias_buf##_conv_##block_number[i] = 0;\
else\
  for (int i = 0; i < FOUT; i++)\
      conv_bias_buf##_conv_##block_number[i] = ((float)(rand()%256 - 128)) / 127.f;\
\
auto input_usr_md##_conv_##block_number = memory::desc(\
    {BS, FIN, N + 2 * already_added_padding, N + 2 * already_added_padding},\
    memory::data_type::f32,\
    memory::format_tag::nchw\
);\
\
auto conv_weights_usr_md##_conv_##block_number = memory::desc(\
    {FOUT, FIN, K, K},\
    memory::data_type::f32,\
    memory::format_tag::oihw\
);\
\
auto conv_bias_usr_md##_conv_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::x\
);\
\
auto conv_weights_usr_mem##_conv_##block_number = memory(conv_weights_usr_md##_conv_##block_number, cpu_engine, conv_weights_buf##_conv_##block_number.data());\
auto conv_bias_usr_mem##_conv_##block_number = memory(conv_bias_usr_md##_conv_##block_number, cpu_engine, conv_bias_buf##_conv_##block_number.data());\
\
auto conv_src_md##_conv_##block_number = memory::desc(\
    {BS, FIN, N + 2 * already_added_padding, N + 2 * already_added_padding},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv_weights_md##_conv_##block_number = memory::desc(\
    {FOUT, FIN, K, K},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv_bias_md##_conv_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto output_md##_conv_##block_number = memory::desc(\
    {BS, FOUT, (N - K + 2 * (already_added_padding + padding)) /stride + 1, (N - K + 2 * (already_added_padding + padding)) /stride + 1},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv_d##_conv_##block_number = convolution_forward::desc(\
    prop_kind::forward_inference,\
    algorithm::convolution_direct,\
    conv_src_md##_conv_##block_number,\
    conv_weights_md##_conv_##block_number,\
    conv_bias_md##_conv_##block_number,\
    output_md##_conv_##block_number,\
    conv_strides##_conv_##block_number,\
    conv_padding##_conv_##block_number,\
    conv_padding##_conv_##block_number\
);\
\
mkldnn::convolution_forward::primitive_desc conv_pd##_conv_##block_number;\
post_ops conv_post_ops##_conv_##block_number;\
primitive_attr conv_attr##_conv_##block_number;\
if(fuse_relu){\
  conv_post_ops##_conv_##block_number.append_eltwise(1, algorithm::eltwise_relu, 0, 0);\
  conv_attr##_conv_##block_number.set_post_ops(conv_post_ops##_conv_##block_number);\
  conv_pd##_conv_##block_number = convolution_forward::primitive_desc(\
    conv_d##_conv_##block_number,\
    conv_attr##_conv_##block_number,\
    cpu_engine\
  );\
}else\
  conv_pd##_conv_##block_number = convolution_forward::primitive_desc(\
    conv_d##_conv_##block_number,\
    cpu_engine\
  );\
\
auto conv_dst_mem##_conv_##block_number = memory(conv_pd##_conv_##block_number.dst_desc(), cpu_engine, output.data());\
\
auto src_desc##_conv_##block_number = conv_pd##_conv_##block_number.src_desc().data.format_desc.blocking;\
auto w_desc##_conv_##block_number = conv_pd##_conv_##block_number.weights_desc().data.format_desc.blocking;\
auto dst_desc##_conv_##block_number = conv_pd##_conv_##block_number.dst_desc().data.format_desc.blocking;\
\
auto input_usr_mem##_conv_##block_number = memory(input_usr_md##_conv_##block_number, cpu_engine, input.data());\
auto input_mem##_conv_##block_number = memory(conv_pd##_conv_##block_number.src_desc(), cpu_engine);\
\
auto conv_weights_mem##_conv_##block_number = conv_weights_usr_mem##_conv_##block_number;\
if (conv_pd##_conv_##block_number.weights_desc() != conv_weights_usr_mem##_conv_##block_number.get_desc()) {\
    conv_weights_mem##_conv_##block_number = memory(conv_pd##_conv_##block_number.weights_desc(), cpu_engine);\
    reorder(conv_weights_usr_mem##_conv_##block_number, conv_weights_mem##_conv_##block_number)\
        .execute(cpu_stream, conv_weights_usr_mem##_conv_##block_number, conv_weights_mem##_conv_##block_number);\
}\
\
auto output_usr_md_conv_##block_number = memory::desc(\
  {BS, FOUT, (N - K + 2 * (already_added_padding + padding)) /stride + 1, (N - K + 2 * (already_added_padding + padding)) /stride + 1},\
  memory::data_type::f32,\
  memory::format_tag::nchw\
);\
auto output_conv_mem_##block_number = memory(output_usr_md_conv_##block_number, cpu_engine);\
auto reshape_conv_output_##block_number = reorder(conv_dst_mem_conv_##block_number, output_conv_mem_##block_number);\
\
net.push_back(convolution_forward(conv_pd##_conv_##block_number));\
net_args.push_back({\
    {MKLDNN_ARG_SRC, input_mem##_conv_##block_number},\
    {MKLDNN_ARG_WEIGHTS, conv_weights_mem##_conv_##block_number},\
    {MKLDNN_ARG_BIAS, conv_bias_usr_mem##_conv_##block_number},\
    {MKLDNN_ARG_DST, conv_dst_mem##_conv_##block_number}\
});\
float* output_conv_mem_handle_##block_number = (float*) output_conv_mem_##block_number.get_data_handle();\
create_add_relu_block_tiramisu(block_number,BS,FOUT,N_OUT,output_resnet.data(),output_conv_mem_handle_##block_number)

#define create_add_relu_block_tiramisu(block_number,BS,FIN,N,output,input1)\
Halide::Buffer<float> arr1_##block_number(input1, N, N, FIN, BS);\
Halide::Buffer<float> output_add_relu_##block_number(output, N, N, FIN, BS);

#define ADD_RELU_TIRAMISU(block_number,BS,FIN,N) \
add_relu_inplace_##BS##_##FIN##_##N##_block(\
    arr1_##block_number.raw_buffer(),\
    output_add_relu_##block_number.raw_buffer()\
);

#define CONV1x1_THEN_ADD_RELU_TIRAMISU(block_number,conv_index,BS,FIN,N)\
reorder(input_usr_mem_conv_##block_number, input_mem_conv_##block_number)\
    .execute(cpu_stream, input_usr_mem_conv_##block_number, input_mem_conv_##block_number);\
net[conv_index].execute(cpu_stream, net_args[conv_index]);\
\
reshape_conv_output_##block_number.execute(cpu_stream, conv_dst_mem_conv_##block_number, output_conv_mem_##block_number);\
ADD_RELU_TIRAMISU(block_number,BS,FIN,N)
