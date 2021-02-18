#define create_sparse_vgg_block_tiramisu(block_number,filename,filename2,BS,FOUT,FIN,N,K,input,output) \
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
\
Halide::Buffer<float> b_conv1_result##_##block_number(N + 2, N + 2, FOUT, BS);\
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
assert((n##_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
b_SIZES##_##block_number(1) = FNNZ2##_##block_number;\
Halide::Buffer<float> b_filter_values2##_##block_number(filter_values2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_idx2##_##block_number(filter_idx2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_finptr2##_##block_number(filter_finptr2##_##block_number, FOUT + 1);\
\
Halide::Buffer<float> b_bias2##_##block_number(FOUT);\
\
Halide::Buffer<float> b_result##_##block_number(output, N / 2 + 2, N / 2 + 2, FOUT, BS);\
\
srand(4);\
\
for (int q=0; q<FOUT; q++)\
  b_bias##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;\
\
for (int q=0; q<FOUT; q++)\
  b_bias2##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;

#define call_sparse_vgg19_block_tiramisu(block_number,function_extension)\
sparse_vgg_block##function_extension(\
  b_SIZES##_##block_number.raw_buffer(),\
  b_input##_##block_number.raw_buffer(),\
    b_filter_values##_##block_number.raw_buffer(),\
    b_filter_idx##_##block_number.raw_buffer(),\
    b_filter_finptr##_##block_number.raw_buffer(),\
    b_bias##_##block_number.raw_buffer(),\
  b_conv1_result##_##block_number.raw_buffer(),\
    b_filter_values2##_##block_number.raw_buffer(),\
    b_filter_idx2##_##block_number.raw_buffer(),\
    b_filter_finptr2##_##block_number.raw_buffer(),\
    b_bias2##_##block_number.raw_buffer(),\
  b_result##_##block_number.raw_buffer()\
);

#define create_sparse_conv_relu_conv_relu_tiramisu(block_number,filename,filename2,BS,FOUT,FIN,N,K,input,output) \
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
\
Halide::Buffer<float> b_conv1_result##_##block_number(N + 2, N + 2, FOUT, BS);\
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
assert((n##_##block_number == N) && ("N parameter specified in configure.h doesn't match the csr weights file's N parameter"));\
\
b_SIZES##_##block_number(1) = FNNZ2##_##block_number;\
Halide::Buffer<float> b_filter_values2##_##block_number(filter_values2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_idx2##_##block_number(filter_idx2##_##block_number, FNNZ2##_##block_number);\
Halide::Buffer<int> b_filter_finptr2##_##block_number(filter_finptr2##_##block_number, FOUT + 1);\
\
Halide::Buffer<float> b_bias2##_##block_number(FOUT);\
\
Halide::Buffer<float> b_result##_##block_number(output, N + 2, N + 2, FOUT, BS);\
\
srand(4);\
\
for (int q=0; q<FOUT; q++)\
  b_bias##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;\
\
srand(4);\
for (int q=0; q<FOUT; q++)\
  b_bias2##_##block_number(q) = ((float)(rand()%256 - 128)) / 127.f;

#define call_sparse_conv_relu_conv_relu_tiramisu(block_number,function_extension)\
sparse_conv_relu_conv_relu##function_extension(\
  b_SIZES##_##block_number.raw_buffer(),\
  b_input##_##block_number.raw_buffer(),\
    b_filter_values##_##block_number.raw_buffer(),\
    b_filter_idx##_##block_number.raw_buffer(),\
    b_filter_finptr##_##block_number.raw_buffer(),\
    b_bias##_##block_number.raw_buffer(),\
  b_conv1_result##_##block_number.raw_buffer(),\
    b_filter_values2##_##block_number.raw_buffer(),\
    b_filter_idx2##_##block_number.raw_buffer(),\
    b_filter_finptr2##_##block_number.raw_buffer(),\
    b_bias2##_##block_number.raw_buffer(),\
  b_result##_##block_number.raw_buffer()\
);
