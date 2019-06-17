#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "configure.h"
#include "mkl_dnn.h"

#define CHECK_ERR(f, err)                                          \
    do                                                             \
    {                                                              \
        (err) = (f);                                               \
        if ((err) != E_SUCCESS)                                    \
        {                                                          \
            printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
            exit(1);                                               \
        }                                                          \
    } while (0)

#define TENSOR_DIMENSION 4

int main()
{
    srand(1);
    dnnError_t err;

    // Define some parameters

    // CONV1
    size_t conv1_input_size[] = {N, N, FIn, BATCH_SIZE};
    size_t conv1_output_size[] = {N, N, FOut, BATCH_SIZE};

    double conv1_filter_param[FOut][FIn][K_Y][K_X];

    size_t conv1_filter_size[] = {K_X, K_Y, FIn, FOut};
    size_t conv1_strides[] = {1, 1};
    int conv1_offset[] = {-1, -1};

    // BN1
    size_t bn1_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t bn1_input_strides[] = {1, N, N*N, N*N*FOut};

    // CONV2
    size_t conv2_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t conv2_output_size[] = {N, N, FOut, BATCH_SIZE};

    double conv2_filter_param[FOut][FOut][K_Y][K_X];

    size_t conv2_filter_size[] = {K_X, K_Y, FOut, FOut};
    size_t conv2_strides[] = {1, 1};
    int conv2_offset[] = {-1, -1};

    // BN2
    size_t bn2_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t bn2_input_strides[] = {1, N, N*N, N*N*FOut};

    double bn_scale_shift[2*FOut];
    double bn_workspace[2*FOut];

    // Initialize parameters
    for (int z = 0; z < FOut; ++z) {
        bn_scale_shift[z] = 1.f;
        bn_scale_shift[z + FOut] = 0.f;
    }

    for (int fout = 0; fout < FOut; ++fout)
        for (int z = 0; z < FIn; ++z)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv1_filter_param[fout][z][k_y][k_x] = 1.f;

    for (int fout = 0; fout < FOut; ++fout)
        for (int z = 0; z < FOut; ++z)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv2_filter_param[fout][z][k_y][k_x] = 1.f;

    // Create the ResNet block (CONV-BN-ReLU-CONV-BN)
    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);
    
    dnnLayout_t bn1_input_layout;
    CHECK_ERR(dnnLayoutCreate_F64(&bn1_input_layout, TENSOR_DIMENSION, bn1_input_size, bn1_input_strides), err);

    dnnPrimitive_t conv1_primitive;
    CHECK_ERR(dnnConvolutionCreateForward_F64(&conv1_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, conv1_input_size, conv1_output_size, conv1_filter_size, conv1_strides, conv1_offset, dnnBorderZeros), err);

    dnnPrimitive_t bn1_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F64(&bn1_primitive, attributes, bn1_input_layout, EPSILON), err);

    dnnPrimitive_t relu_primitive;
    CHECK_ERR(dnnReLUCreateForward_F64(&relu_primitive, attributes, bn1_input_layout, 0.f), err);

    dnnPrimitive_t conv2_primitive;
    CHECK_ERR(dnnConvolutionCreateForward_F64(&conv2_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, conv2_input_size, conv2_output_size, conv2_filter_size, conv2_strides, conv2_offset, dnnBorderZeros), err);

    dnnLayout_t bn2_input_layout;
    CHECK_ERR(dnnLayoutCreate_F64(&bn2_input_layout, TENSOR_DIMENSION, bn2_input_size, bn2_input_strides), err);

    dnnPrimitive_t bn2_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F64(&bn2_primitive, attributes, bn2_input_layout, EPSILON), err);

    // Allocate buffers
    double* input_buf = malloc(sizeof(double) * N * N * FIn * BATCH_SIZE);
    double* tmp_buf = malloc(sizeof(double) * N * N * FOut * BATCH_SIZE);
    double* output_buf = malloc(sizeof(double) * N * N * FOut * BATCH_SIZE);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*FIn] = (double)(rand() % 1000);

    // Execute the block
    double* resources[dnnResourceNumber] = {0};
    double times[NB_TESTS];
    clock_t start, end;

    for (int i = 0; i < NB_TESTS; ++i) {
        start = clock();

        resources[dnnResourceSrc] = input_buf;
        resources[dnnResourceFilter] = (void*)conv1_filter_param;
        resources[dnnResourceDst] = tmp_buf;
        CHECK_ERR(dnnExecute_F64(conv1_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = tmp_buf;
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = tmp_buf;
        CHECK_ERR(dnnExecute_F64(bn1_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = tmp_buf;
        resources[dnnResourceDst] = tmp_buf;
        CHECK_ERR(dnnExecute_F64(relu_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = tmp_buf;
        resources[dnnResourceFilter] = (void*)conv2_filter_param;
        resources[dnnResourceDst] = output_buf;
        CHECK_ERR(dnnExecute_F64(conv2_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = output_buf;
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = output_buf;
        CHECK_ERR(dnnExecute_F64(bn2_primitive, (void**)resources), err);

        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }

    printf("\n\n\tMKL ResNet block duration : %f ms.\n", median(NB_TESTS, times));

    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 1;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%g\n", (float)((int)(output_buf[x + y*N + z*N*N + n*N*N*FOut] * 100) / 100.0));

    fclose(f);
    
    free(output_buf);
    free(tmp_buf);
    free(input_buf);
    dnnDelete_F64(bn2_primitive);
    dnnLayoutDelete_F64(bn2_input_layout);
    dnnDelete_F64(conv2_primitive);
    dnnDelete_F64(relu_primitive);
    dnnDelete_F64(bn1_primitive);
    dnnDelete_F64(conv1_primitive);
    dnnPrimitiveAttributesDestroy_F64(attributes);
    dnnLayoutDelete_F64(bn1_input_layout);
    
    return 0;
}
