#define create_and_add_vgg_block_mkldnn(block_number,filename,filename2,BS,FOUT,FIN,N,K,RESIZE_INPUT,input,interm,output, PREVIOUS_PD) \
memory::dims conv_strides##_##block_number = {1, 1};\
memory::dims conv1_padding##_##block_number = {0, 0};\
memory::dims conv2_padding##_##block_number = {1, 1};\
\
memory::dims pool_strides##_##block_number = {2, 2};\
memory::dims pool_kernel##_##block_number = {2, 2};\
memory::dims pool_padding##_##block_number = {0, 0};\
\
std::vector<float> conv1_bias_buf##_##block_number(FOUT);\
\
std::vector<float> conv2_bias_buf##_##block_number(FOUT);\
\
int used_FOUT##_##block_number;\
int used_FIN##_##block_number;\
int used_K##_##block_number;\
int n##_##block_number;\
float* weights1_buf##_##block_number;\
importCSRFromFileAsDense(filename, &weights1_buf##_##block_number, &used_FOUT##_##block_number, &used_FIN##_##block_number, &used_K##_##block_number, &n##_##block_number);\
\
assert((used_FOUT##_##block_number == FOUT) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_##block_number == FIN) && ("FIn parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
float* weights2_buf##_##block_number;\
importCSRFromFileAsDense(filename2, &weights2_buf##_##block_number, &used_FOUT##_##block_number, &used_FIN##_##block_number, &used_K##_##block_number, &n##_##block_number);\
\
assert((used_FOUT##_##block_number == FOUT) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FOUT parameter."));\
assert((used_FIN##_##block_number == FOUT) && ("FOut parameter specified in configure.h doesn't match the csr weights file's FIn parameter"));\
assert((used_K##_##block_number == K) && ("K parameter specified in configure.h doesn't match the csr weights file's K parameter"));\
assert((n##_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
std::vector<float> conv1_weights_buf##_##block_number(weights1_buf##_##block_number, weights1_buf##_##block_number + FOUT*FIN*K*K);\
std::vector<float> conv2_weights_buf##_##block_number(weights2_buf##_##block_number, weights2_buf##_##block_number + FOUT*FOUT*K*K);\
\
srand(4);\
for (int i = 0; i < FOUT; i++)\
    conv1_bias_buf##_##block_number[i] = ((float)(rand()%256 - 128)) / 127.f;\
\
for (int i = 0; i < FOUT; i++)\
    conv2_bias_buf##_##block_number[i] = ((float)(rand()%256 - 128)) / 127.f;\
\
auto conv1_weights_usr_md##_##block_number = memory::desc(\
    {FOUT, FIN, K, K},\
    memory::data_type::f32,\
    memory::format_tag::oihw\
);\
\
auto conv1_bias_usr_md##_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::x\
);\
\
auto conv1_weights_usr_mem##_##block_number = memory(conv1_weights_usr_md##_##block_number, cpu_engine, conv1_weights_buf##_##block_number.data());\
auto conv1_bias_usr_mem##_##block_number = memory(conv1_bias_usr_md##_##block_number, cpu_engine, conv1_bias_buf##_##block_number.data());\
\
auto input_usr_md##_##block_number = memory::desc(\
    {BS, FIN, N + 2, N + 2},\
    memory::data_type::f32,\
    memory::format_tag::nchw\
);\
\
auto conv1_src_md##_##block_number = memory::desc(\
    {BS, FIN, N + 2, N + 2},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv1_weights_md##_##block_number = memory::desc(\
    {FOUT, FIN, K, K},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv1_bias_md##_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto output1_md##_##block_number = memory::desc(\
    {BS, FOUT, N, N},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv1_d##_##block_number = convolution_forward::desc(\
    prop_kind::forward_inference,\
    algorithm::convolution_direct,\
    conv1_src_md##_##block_number,\
    conv1_weights_md##_##block_number,\
    conv1_bias_md##_##block_number,\
    output1_md##_##block_number,\
    conv_strides##_##block_number,\
    conv1_padding##_##block_number,\
    conv1_padding##_##block_number\
);\
\
post_ops conv1_post_ops##_##block_number;\
conv1_post_ops##_##block_number.append_eltwise(1, algorithm::eltwise_relu, 0, 0);\
\
primitive_attr conv1_attr##_##block_number;\
conv1_attr##_##block_number.set_post_ops(conv1_post_ops##_##block_number);\
\
auto conv1_pd##_##block_number = convolution_forward::primitive_desc(\
    conv1_d##_##block_number,\
    conv1_attr##_##block_number,\
    cpu_engine\
);\
\
auto conv1_dst_mem##_##block_number = memory(conv1_pd##_##block_number.dst_desc(), cpu_engine, interm.data());\
\
mkldnn::memory input_usr_mem##_##block_number;\
if (RESIZE_INPUT)\
  input_usr_mem##_##block_number = memory(input_usr_md##_##block_number, cpu_engine, input.data());\
else\
  input_usr_mem##_##block_number = memory(PREVIOUS_PD.dst_desc(), cpu_engine, input.data());\
\
auto input_mem##_##block_number = memory(conv1_pd##_##block_number.src_desc(), cpu_engine);\
\
if (RESIZE_INPUT)\
  reorder(input_usr_mem##_##block_number, input_mem##_##block_number)\
    .execute(cpu_stream, input_usr_mem##_##block_number, input_mem##_##block_number);\
\
auto conv1_weights_mem##_##block_number = conv1_weights_usr_mem##_##block_number;\
if (conv1_pd##_##block_number.weights_desc() != conv1_weights_usr_mem##_##block_number.get_desc()) {\
    conv1_weights_mem##_##block_number = memory(conv1_pd##_##block_number.weights_desc(), cpu_engine);\
    reorder(conv1_weights_usr_mem##_##block_number, conv1_weights_mem##_##block_number)\
        .execute(cpu_stream, conv1_weights_usr_mem##_##block_number, conv1_weights_mem##_##block_number);\
}\
\
net.push_back(convolution_forward(conv1_pd##_##block_number));\
if (RESIZE_INPUT)\
  net_args.push_back({\
      {MKLDNN_ARG_SRC, input_mem##_##block_number},\
      {MKLDNN_ARG_WEIGHTS, conv1_weights_mem##_##block_number},\
      {MKLDNN_ARG_BIAS, conv1_bias_usr_mem##_##block_number},\
      {MKLDNN_ARG_DST, conv1_dst_mem##_##block_number}\
  });\
else\
  net_args.push_back({\
      {MKLDNN_ARG_SRC, input_usr_mem##_##block_number},\
      {MKLDNN_ARG_WEIGHTS, conv1_weights_mem##_##block_number},\
      {MKLDNN_ARG_BIAS, conv1_bias_usr_mem##_##block_number},\
      {MKLDNN_ARG_DST, conv1_dst_mem##_##block_number}\
  });\
\
auto conv2_weights_usr_md##_##block_number = memory::desc(\
    {FOUT, FOUT, K, K},\
    memory::data_type::f32,\
    memory::format_tag::oihw\
);\
\
auto conv2_bias_usr_md##_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::x\
);\
\
auto conv2_weights_usr_mem##_##block_number = memory(conv2_weights_usr_md##_##block_number, cpu_engine, conv2_weights_buf##_##block_number.data());\
auto conv2_bias_usr_mem##_##block_number = memory(conv2_bias_usr_md##_##block_number, cpu_engine, conv2_bias_buf##_##block_number.data());\
\
auto conv2_weights_md##_##block_number = memory::desc(\
    {FOUT, FOUT, K, K},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv2_bias_md##_##block_number = memory::desc(\
    {FOUT},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto output2_md##_##block_number = memory::desc(\
    {BS, FOUT, N, N},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto conv2_d##_##block_number = convolution_forward::desc(\
    prop_kind::forward_inference,\
    algorithm::convolution_direct,\
    conv1_pd##_##block_number.dst_desc(),\
    conv2_weights_md##_##block_number,\
    conv2_bias_md##_##block_number,\
    output2_md##_##block_number,\
    conv_strides##_##block_number,\
    conv2_padding##_##block_number,\
    conv2_padding##_##block_number\
);\
\
auto conv2_pd##_##block_number = convolution_forward::primitive_desc(\
    conv2_d##_##block_number,\
    conv1_attr##_##block_number,\
    cpu_engine\
);\
\
auto conv2_dst_mem##_##block_number = memory(conv2_pd##_##block_number.dst_desc(), cpu_engine);\
\
auto conv2_weights_mem##_##block_number = conv2_weights_usr_mem##_##block_number;\
if (conv2_pd##_##block_number.weights_desc() != conv2_weights_usr_mem##_##block_number.get_desc()) {\
    conv2_weights_mem##_##block_number = memory(conv2_pd##_##block_number.weights_desc(), cpu_engine);\
    reorder(conv2_weights_usr_mem##_##block_number, conv2_weights_mem##_##block_number)\
        .execute(cpu_stream, conv2_weights_usr_mem##_##block_number, conv2_weights_mem##_##block_number);\
}\
\
net.push_back(convolution_forward(conv2_pd##_##block_number));\
net_args.push_back({\
    {MKLDNN_ARG_SRC, conv1_dst_mem##_##block_number},\
    {MKLDNN_ARG_WEIGHTS, conv2_weights_mem##_##block_number},\
    {MKLDNN_ARG_BIAS, conv2_bias_usr_mem##_##block_number},\
    {MKLDNN_ARG_DST, conv2_dst_mem##_##block_number}\
});\
\
auto pool_output_md##_##block_number = memory::desc(\
    {BS, FOUT, N/2, N/2},\
    memory::data_type::f32,\
    memory::format_tag::any\
);\
\
auto pool_d##_##block_number = pooling_forward::desc(\
    prop_kind::forward_inference,\
    algorithm::pooling_max,\
    conv2_pd##_##block_number.dst_desc(),\
    pool_output_md##_##block_number,\
    pool_strides##_##block_number,\
    pool_kernel##_##block_number,\
    pool_padding##_##block_number,\
    pool_padding##_##block_number\
);\
\
auto pool_pd##_##block_number = pooling_forward::primitive_desc(\
    pool_d##_##block_number,\
    cpu_engine\
);\
\
auto pool_dst_mem##_##block_number = memory(pool_pd##_##block_number.dst_desc(), cpu_engine);\
\
net.push_back(pooling_forward(pool_pd##_##block_number));\
net_args.push_back({\
    {MKLDNN_ARG_SRC, conv2_dst_mem##_##block_number},\
    {MKLDNN_ARG_DST, pool_dst_mem##_##block_number}\
});

#define convolution_mkldnn(block_number,import_from_file,filename,density,BS,FOUT,FIN,N,K,stride,padding,already_added_padding,input,output,fuse_relu,PREVIOUS_PD)\
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
  srand(4);\
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
srand(4);\
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
auto input_usr_mem##_conv_##block_number = memory(PREVIOUS_PD.dst_desc(), cpu_engine, input.data());\
auto conv_weights_mem##_conv_##block_number = conv_weights_usr_mem##_conv_##block_number;\
if (conv_pd##_conv_##block_number.weights_desc() != conv_weights_usr_mem##_conv_##block_number.get_desc()) {\
    conv_weights_mem##_conv_##block_number = memory(conv_pd##_conv_##block_number.weights_desc(), cpu_engine);\
    reorder(conv_weights_usr_mem##_conv_##block_number, conv_weights_mem##_conv_##block_number)\
        .execute(cpu_stream, conv_weights_usr_mem##_conv_##block_number, conv_weights_mem##_conv_##block_number);\
}\
\
net.push_back(convolution_forward(conv_pd##_conv_##block_number));\
net_args.push_back({\
    {MKLDNN_ARG_SRC, input_usr_mem##_conv_##block_number},\
    {MKLDNN_ARG_WEIGHTS, conv_weights_mem##_conv_##block_number},\
    {MKLDNN_ARG_BIAS, conv_bias_usr_mem##_conv_##block_number},\
    {MKLDNN_ARG_DST, conv_dst_mem##_conv_##block_number}\
});
