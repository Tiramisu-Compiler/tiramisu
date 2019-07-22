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

static dnnError_t init_conversion(dnnPrimitive_t *cv, float **ptr_out,
                                  dnnLayout_t lt_pr, dnnLayout_t lt_us, float *ptr_us)
{
    dnnError_t err;
    *ptr_out = NULL;
    if (!dnnLayoutCompare_F32(lt_pr, lt_us))
    {
        CHECK_ERR(dnnConversionCreate_F32(cv, lt_us, lt_pr), err);
        CHECK_ERR(dnnAllocateBuffer_F32((void **)ptr_out, lt_pr), err);
    }
    else
    {
        *ptr_out = ptr_us;
    }
    return E_SUCCESS;

bail_out:
    if (*ptr_out)
        dnnReleaseBuffer_F32(*ptr_out);
    return err;
}

int main()
{
    srand(1);
    dnnError_t err;

    // Define some parameters

    // CONV1
    size_t conv1_input_size[] = {N, N, FIn, BATCH_SIZE};
    size_t conv1_input_strides[] = {1, N, N*N, N*N*FIn};

    float conv1_filter_param[FOut][FIn][K_Y][K_X];

    size_t conv1_filter_size[] = {K_X, K_Y, FIn, FOut};
    size_t conv1_filter_strides[] = {1, K_X, K_X*K_Y, K_X*K_Y*FIn};

    size_t conv1_strides[] = {1, 1};
    int conv1_offset[] = {-1, -1};

    // BN1
    size_t bn1_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t bn1_input_strides[] = {1, N, N*N, N*N*FOut};

    // CONV2
    size_t conv2_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t conv2_output_size[] = {N, N, FOut, BATCH_SIZE};

    float conv2_filter_param[FOut][FOut][K_Y][K_X];

    size_t conv2_filter_size[] = {K_X, K_Y, FOut, FOut};
    size_t conv2_filter_strides[] = {1, K_X, K_X*K_Y, K_X*K_Y*FOut};

    size_t conv2_strides[] = {1, 1};
    int conv2_offset[] = {-1, -1};

    // BN2
    size_t bn2_input_size[] = {N, N, FOut, BATCH_SIZE};
    size_t bn2_input_strides[] = {1, N, N*N, N*N*FOut};

    size_t output_size[] = {N, N, FOut, BATCH_SIZE};
    size_t output_strides[] = {1, N, N*N, N*N*FOut};

    float bn_scale_shift[2*FOut];
    float bn_workspace[2*FOut];

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

    // Allocate buffers
    float* input_buf = malloc(sizeof(float) * N * N * FIn * BATCH_SIZE);
    float* output_buf = malloc(sizeof(float) * N * N * FOut * BATCH_SIZE);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*FIn] = (float)(rand() % 1000);

    // Create the ResNet block (CONV-BN-ReLU-CONV-BN)
    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F32(&attributes), err);

    // Create convolution 1
    dnnLayout_t lt_user_input, lt_user_filt1, lt_user_output;
    dnnLayout_t lt_conv1_input, lt_conv1_filt, lt_conv1_output;

    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_input, TENSOR_DIMENSION, conv1_input_size, conv1_input_strides), err);
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt1, TENSOR_DIMENSION, conv1_filter_size, conv1_filter_strides), err);
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_output, TENSOR_DIMENSION, output_size, output_strides), err);

    float* res_conv1[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_usr_to_conv1_input, cv_usr_to_conv1_filt;

    dnnPrimitive_t conv1_primitive;
    CHECK_ERR(dnnConvolutionCreateForward_F32(&conv1_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, conv1_input_size, output_size, conv1_filter_size, conv1_strides, conv1_offset, dnnBorderZeros), err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv1_primitive, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv1_primitive, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output, conv1_primitive, dnnResourceDst), err);

    CHECK_ERR(init_conversion(&cv_usr_to_conv1_input, &res_conv1[dnnResourceSrc], lt_conv1_input, lt_user_input, input_buf), err);
    CHECK_ERR(init_conversion(&cv_usr_to_conv1_filt, &res_conv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt1, (void*)conv1_filter_param), err);
    CHECK_ERR(dnnAllocateBuffer_F32((void **)&res_conv1[dnnResourceDst], lt_conv1_output), err);

    if (cv_usr_to_conv1_filt)
        CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv1_filt, (void*)conv1_filter_param, res_conv1[dnnResourceFilter]), err);
    if (cv_usr_to_conv1_input)
        CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv1_input, input_buf, res_conv1[dnnResourceSrc]), err);

    // Create BN 1
    dnnPrimitive_t bn1_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F32(&bn1_primitive, attributes, lt_conv1_output, EPSILON), err);

    // Create ReLU
    dnnPrimitive_t relu_primitive;
    CHECK_ERR(dnnReLUCreateForward_F32(&relu_primitive, attributes, lt_conv1_output, 0.f), err);

    // Create convolution 2
    dnnLayout_t lt_user_filt2;
    dnnLayout_t lt_conv2_filt, lt_conv2_output;

    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt2, TENSOR_DIMENSION, conv2_filter_size, conv2_filter_strides), err);

    float* res_conv2[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_usr_to_conv2_filt;

    dnnPrimitive_t conv2_primitive;
    CHECK_ERR(dnnConvolutionCreateForward_F32(&conv2_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, conv2_input_size, conv2_output_size, conv2_filter_size, conv2_strides, conv2_offset, dnnBorderZeros), err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv2_filt, conv2_primitive, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv2_output, conv2_primitive, dnnResourceDst), err);

    CHECK_ERR(init_conversion(&cv_usr_to_conv2_filt, &res_conv2[dnnResourceFilter], lt_conv2_filt, lt_user_filt2, (void*)conv2_filter_param), err);
    CHECK_ERR(dnnAllocateBuffer_F32((void **)&res_conv2[dnnResourceDst], lt_conv2_output), err);

    if (cv_usr_to_conv2_filt)
        CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv2_filt, (void*)conv2_filter_param, res_conv2[dnnResourceFilter]), err);

    // Create BN 2
    dnnPrimitive_t bn2_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F32(&bn2_primitive, attributes, lt_conv2_output, EPSILON), err);

    // Create output conversions
    dnnPrimitive_t cv_bn2_to_usr_output;
    CHECK_ERR(init_conversion(&cv_bn2_to_usr_output, &output_buf, lt_user_output, lt_conv2_output, res_conv2[dnnResourceDst]), err);

    // Execute the block
    float* resources[dnnResourceNumber] = {0};
    double times[NB_TESTS];
    clock_t start, end;

    for (int i = 0; i < NB_TESTS; ++i) {
        start = clock();
        
        CHECK_ERR(dnnExecute_F32(conv1_primitive, (void**)res_conv1), err);

        resources[dnnResourceSrc] = res_conv1[dnnResourceDst];
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = res_conv1[dnnResourceDst];
        CHECK_ERR(dnnExecute_F32(bn1_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = res_conv1[dnnResourceDst];
        resources[dnnResourceDst] = res_conv1[dnnResourceDst];
        CHECK_ERR(dnnExecute_F32(relu_primitive, (void**)resources), err);

        res_conv2[dnnResourceSrc] = res_conv1[dnnResourceDst];
        CHECK_ERR(dnnExecute_F32(conv2_primitive, (void**)res_conv2), err);

        resources[dnnResourceSrc] = res_conv2[dnnResourceDst];
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = res_conv2[dnnResourceDst];
        CHECK_ERR(dnnExecute_F32(bn2_primitive, (void**)resources), err);

        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }

    printf("\n\n\tMKL ResNet block duration : %f ms.\n", median(NB_TESTS, times));

    if (cv_bn2_to_usr_output)
        CHECK_ERR(dnnConversionExecute_F32(cv_bn2_to_usr_output, res_conv2[dnnResourceDst], output_buf), err);

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
                    fprintf(f, "%.10g\n", output_buf[x + y*N + z*N*N + n*N*N*FOut]);

    fclose(f);

    free(input_buf);

    dnnDelete_F32(bn2_primitive);
    dnnDelete_F32(conv2_primitive);
    dnnDelete_F32(relu_primitive);
    dnnDelete_F32(bn1_primitive);
    dnnDelete_F32(conv1_primitive);
    dnnDelete_F32(cv_usr_to_conv1_filt);
    dnnDelete_F32(cv_usr_to_conv1_input);
    dnnDelete_F32(cv_usr_to_conv2_filt);
    dnnDelete_F32(cv_bn2_to_usr_output);

    dnnPrimitiveAttributesDestroy_F32(attributes);
    dnnLayoutDelete_F32(lt_user_input);
    dnnLayoutDelete_F32(lt_user_filt1);
    dnnLayoutDelete_F32(lt_user_filt2);
    dnnLayoutDelete_F32(lt_user_output);
    dnnLayoutDelete_F32(lt_conv1_input);
    dnnLayoutDelete_F32(lt_conv1_filt);
    dnnLayoutDelete_F32(lt_conv1_output);
    dnnLayoutDelete_F32(lt_conv2_filt);
    dnnLayoutDelete_F32(lt_conv2_output);

    dnnReleaseBuffer_F32(res_conv1[dnnResourceSrc]);
    dnnReleaseBuffer_F32(res_conv1[dnnResourceFilter]);
    dnnReleaseBuffer_F32(res_conv2[dnnResourceFilter]);
    dnnReleaseBuffer_F32(res_conv1[dnnResourceDst]);
    dnnReleaseBuffer_F32(res_conv2[dnnResourceDst]);
    
    return 0;
}
