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
    size_t input_size[] = {N+2, N+2, 4*GR, BATCH_SIZE};
    size_t input_strides[] = {1, N+2, (N+2)*(N+2), (N+2)*(N+2)*4*GR};

    size_t output_size[] = {N, N, GR, BATCH_SIZE};
    size_t output_strides[] = {1, N, N*N, N*N*GR};

    float bn_scale_shift[2*4*GR];
    float bn_workspace[2*4*GR];

    float conv_filter_param[GR][4*GR][K_Y][K_X];
    float conv_bias_param[GR];

    size_t conv_filter_size[] = {K_X, K_Y, 4*GR, GR};
    size_t conv_filter_strides[] = {1, K_X, K_X*K_Y, K_X*K_Y*4*GR};

    size_t conv_strides[] = {1, 1};
    int conv_offset[] = {0, 0};

    for (int fin = 0; fin < 4*GR; ++fin) {
        bn_scale_shift[fin] = ((float)(rand()%256)) / 255.f;
        if (bn_scale_shift[fin] == 0.f)
            bn_scale_shift[fin] = 1.f;

        bn_scale_shift[fin + 4*GR] = ((float)(rand()%256 - 128)) / 127.f;
    }

    for (int fout = 0; fout < GR; ++fout)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter_param[fout][fin][k_y][k_x] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias_param[fout] = ((float)(rand()%256 - 128)) / 127.f;

    // Allocate buffers
    float* input_buf = malloc(sizeof(float) * (N+2) * (N+2) * 4*GR * BATCH_SIZE);
    float* output_buf = malloc(sizeof(float) * N * N * GR * BATCH_SIZE);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int y = 0; y < N + 2; ++y)
                for (int x = 0; x < N + 2; ++x)
                    input_buf[x + y*(N+2) + fin*(N+2)*(N+2) + n*(N+2)*(N+2)*4*GR] = ((float)(rand()%256 - 128)) / 127.f;

    // Create the DenseNet block
    float* res_bn_relu[dnnResourceNumber] = {0};
    float* res_conv[dnnResourceNumber] = {0};

    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    // Create convolution (begin by convolution to get optimized data layout)
    dnnLayout_t lt_user_input, lt_user_filt, lt_user_output;
    dnnLayout_t lt_conv_input, lt_conv_filt, lt_conv_output;

    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_input, TENSOR_DIMENSION, input_size, input_strides), err);
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt, TENSOR_DIMENSION, conv_filter_size, conv_filter_strides), err);
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_output, TENSOR_DIMENSION, output_size, output_strides), err);

    dnnPrimitive_t cv_usr_to_conv_input, cv_usr_to_conv_filt;
    dnnPrimitive_t conv_primitive;
    CHECK_ERR(dnnConvolutionCreateForwardBias_F32(&conv_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, input_size, output_size, conv_filter_size, conv_strides, conv_offset, dnnBorderZeros), err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_input, conv_primitive, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_filt, conv_primitive, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_output, conv_primitive, dnnResourceDst), err);

    CHECK_ERR(init_conversion(&cv_usr_to_conv_input, &res_bn_relu[dnnResourceSrc], lt_conv_input, lt_user_input, input_buf), err);
    CHECK_ERR(init_conversion(&cv_usr_to_conv_filt, &res_conv[dnnResourceFilter], lt_conv_filt, lt_user_filt, (void*)conv_filter_param), err);
    CHECK_ERR(dnnAllocateBuffer_F32((void **)&res_conv[dnnResourceDst], lt_conv_output), err);

    CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv_filt, (void*)conv_filter_param, res_conv[dnnResourceFilter]), err);

    res_conv[dnnResourceSrc] = res_bn_relu[dnnResourceSrc];
    res_conv[dnnResourceBias] = conv_bias_param;

    // Create BN and ReLU
    dnnPrimitive_t bn_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F32(&bn_primitive, attributes, lt_conv_input, EPSILON), err);

    dnnPrimitive_t relu_primitive;
    CHECK_ERR(dnnReLUCreateForward_F32(&relu_primitive, attributes, lt_conv_input, 0.f), err);

    res_bn_relu[dnnResourceDst] = res_bn_relu[dnnResourceSrc];
    res_bn_relu[dnnResourceScaleShift] = bn_scale_shift;
    res_bn_relu[dnnResourceWorkspace] = bn_workspace;

    // Create output conversions
    dnnPrimitive_t cv_conv_to_usr_output;
    CHECK_ERR(init_conversion(&cv_conv_to_usr_output, &output_buf, lt_user_output, lt_conv_output, res_conv[dnnResourceDst]), err);

    // Execute the block
    double times[NB_TESTS];
    double start, end;

    for (int i = 0; i < NB_TESTS; ++i) {
        CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv_input, input_buf, res_bn_relu[dnnResourceSrc]), err);
        
        start = rtclock();

        CHECK_ERR(dnnExecute_F32(bn_primitive, (void**)res_bn_relu), err);
        CHECK_ERR(dnnExecute_F32(relu_primitive, (void**)res_bn_relu), err);
        CHECK_ERR(dnnExecute_F32(conv_primitive, (void**)res_conv), err);

        end = rtclock();
        times[i] = (end - start) * 1000;
    }

    printf("\n\n\tDenseNet block time: %f ms.\n", median(NB_TESTS, times));

    CHECK_ERR(dnnConversionExecute_F32(cv_conv_to_usr_output, res_conv[dnnResourceDst], output_buf), err);

    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 1;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output_buf[x + y*N + z*N*N + n*N*N*GR]);

    fclose(f);
    
    free(input_buf);
    dnnDelete_F64(conv_primitive);
    dnnDelete_F64(relu_primitive);
    dnnDelete_F64(bn_primitive);

    dnnDelete_F32(cv_usr_to_conv_filt);
    dnnDelete_F32(cv_usr_to_conv_input);
    dnnDelete_F32(cv_conv_to_usr_output);

    dnnPrimitiveAttributesDestroy_F64(attributes);
    dnnLayoutDelete_F32(lt_user_input);
    dnnLayoutDelete_F32(lt_user_filt);
    dnnLayoutDelete_F32(lt_user_output);
    dnnLayoutDelete_F32(lt_conv_input);
    dnnLayoutDelete_F32(lt_conv_filt);
    dnnLayoutDelete_F32(lt_conv_output);

    dnnReleaseBuffer_F32(res_bn_relu[dnnResourceSrc]);
    dnnReleaseBuffer_F32(res_conv[dnnResourceFilter]);
    dnnReleaseBuffer_F32(res_conv[dnnResourceDst]);

    return 0;
}
