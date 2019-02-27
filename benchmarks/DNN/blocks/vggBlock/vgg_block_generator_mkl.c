#include <stdio.h>
#include <stdlib.h>
#include "configure.h"
#include <time.h>
#include "mkl_dnn.h"

#define CHECK_ERR(f, err)                                          \
    do                                                             \
    {                                                              \
        (err) = (f);                                               \
        if ((err) != E_SUCCESS)                                    \
        {                                                          \
            printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
            goto bail_out;                                         \
        }                                                          \
    } while (0)

#define dimension (4)

static dnnError_t init_conversion(dnnPrimitive_t *cv, double **ptr_out,
                                  dnnLayout_t lt_pr, dnnLayout_t lt_us, double *ptr_us)
{
    dnnError_t err;
    *ptr_out = NULL;
    if (!dnnLayoutCompare_F64(lt_pr, lt_us))
    {
        CHECK_ERR(dnnConversionCreate_F64(cv, lt_us, lt_pr), err);
        CHECK_ERR(dnnAllocateBuffer_F64((void **)ptr_out, lt_pr), err);
    }
    else
    {
        *ptr_out = ptr_us;
    }
    return E_SUCCESS;

bail_out:
    if (*ptr_out)
        dnnReleaseBuffer_F64(*ptr_out);
    return err;
}

static dnnError_t simple_net(int want_groups_conv)
{
    dnnError_t err;

    size_t output2Size[dimension] = {(N - K), (N - K), FOut, BATCH_SIZE};
    size_t outputSize[dimension] = {N, N, FOut, BATCH_SIZE};
    size_t inputSize[dimension] = {N + K, N + K, FIn, BATCH_SIZE};
    size_t inputStrides[dimension] = {1, N + K, (N + K) * (N + K), (N + K) * (N + K) * FIn};
    size_t filter2Size[dimension] = {(K + 1), (K + 1), FOut, FOut};
    size_t filter2Strides[dimension] = {1, (K + 1), (K + 1) * (K + 1), (K + 1) * (K + 1) * FOut};
    size_t filterSize[dimension] = {(K + 1), (K + 1), FIn, FOut};
    size_t filterStrides[dimension] = {1, (K + 1), (K + 1) * (K + 1), (K + 1) * (K + 1) * FIn};
    size_t convolutionStride[dimension - 2] = {1, 1};
    int inputOffset[dimension - 2] = {0, 0};
    size_t biasSize[1] = {outputSize[2]};
    size_t biasStrides[1] = {1};
    size_t pool_outputSize[dimension] = {(N - 2 * K), (N - 2 * K), FOut, BATCH_SIZE};
    size_t pool_outputStrides[dimension] = {1, (N - 2 * K), (N - 2 * K) * (N - 2 * K), (N - 2 * K) * (N - 2 * K) * FOut};
    size_t kernelSize[2] = {(K + 1), (K + 1)};
    size_t kernelStride[2] = {1, 1};

    dnnLayout_t lt_user_input = NULL,
                lt_user_filt = NULL,
                lt_user_bias = NULL,
                lt_user_output = NULL,
                lt_user_filt2 = NULL;
    dnnPrimitive_t conv1 = NULL;
    dnnLayout_t lt_conv1_input = NULL,
                lt_conv1_filt = NULL,
                lt_conv1_bias = NULL,
                lt_conv1_output = NULL;
    double *resConv1[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_user_to_conv1_input = NULL,
                   cv_user_to_conv1_filt = NULL,
                   cv_user_to_conv1_bias = NULL;
    dnnPrimitive_t conv2 = NULL;
    dnnLayout_t lt_conv2_filt = NULL,
                lt_conv2_output = NULL;
    double *resConv2[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_user_to_conv2_filt = NULL;
    dnnPrimitive_t relu1 = NULL;
    double *resRelu1[dnnResourceNumber] = {0};
    dnnLayout_t lt_relu1_output = NULL;
    dnnPrimitive_t relu2 = NULL;
    double *resRelu2[dnnResourceNumber] = {0};
    dnnLayout_t lt_relu2_output = NULL;
    dnnLayout_t lt_pool1_input = NULL;
    dnnPrimitive_t pool1 = NULL;
    void *resPool1[dnnResourceNumber] = {0};
    dnnLayout_t lt_pool1_output = NULL,
                lt_pool1_workspace = NULL;
    dnnPrimitive_t cv_pool1_to_user_output = NULL;
    dnnPrimitiveAttributes_t attributes = NULL;

    double *user_i = NULL,
           *user_c1_f = NULL,
           *user_c1_b = NULL,
           *user_c2_f = NULL,
           *user_o = NULL;

    /*** Data allocation ***/
    user_i = (double *)malloc(sizeof(double) * (inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]));
    user_c1_f = (double *)malloc(sizeof(double) * (filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]));
    user_c2_f = (double *)malloc(sizeof(double) * (filter2Size[0] * filter2Size[1] * filter2Size[2] * filter2Size[3]));
    user_c1_b = (double *)malloc(sizeof(double) * (outputSize[2]));
    if (user_i == NULL || user_c1_f == NULL || user_c2_f == NULL || user_c1_b == NULL)
    {
        err = E_MEMORY_ERROR;
        goto bail_out;
    }

    /*** User's data description ***/
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_input, dimension, inputSize, inputStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_filt, dimension, filterSize, filterStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_bias, 1, biasSize, biasStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_output, dimension, pool_outputSize, pool_outputStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_filt2, dimension, filter2Size, filter2Strides), err);

    /* Initialize attributes */
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    /*** Convolution section ***/
    if (!want_groups_conv)
    {
        CHECK_ERR(dnnConvolutionCreateForwardBias_F64(&conv1, attributes,
                                                      dnnAlgorithmConvolutionDirect, dimension, inputSize,
                                                      outputSize, filterSize, convolutionStride, inputOffset,
                                                      dnnBorderZeros),
                  err);
    }
    else
    {
        CHECK_ERR(dnnGroupsConvolutionCreateForwardBias_F64(&conv1, attributes,
                                                            dnnAlgorithmConvolutionDirect, 1, dimension, inputSize,
                                                            outputSize, filterSize, convolutionStride, inputOffset,
                                                            dnnBorderZeros),
                  err);
    }

    // Convolution describes what layout it expects
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv1_input, conv1, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv1_filt, conv1, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv1_bias, conv1, dnnResourceBias), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv1_output, conv1, dnnResourceDst), err);

    CHECK_ERR(init_conversion(&cv_user_to_conv1_input, &resConv1[dnnResourceSrc], lt_conv1_input, lt_user_input, user_i), err);
    CHECK_ERR(init_conversion(&cv_user_to_conv1_filt, &resConv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt, user_c1_f), err);
    CHECK_ERR(init_conversion(&cv_user_to_conv1_bias, &resConv1[dnnResourceBias], lt_conv1_bias, lt_user_bias, user_c1_b), err);
    CHECK_ERR(dnnAllocateBuffer_F64((void **)&resConv1[dnnResourceDst], lt_conv1_output), err);

    /*** ReLU section ***/
    CHECK_ERR(dnnReLUCreateForward_F64(&relu1, attributes, lt_conv1_output, 0.0f), err);
    resRelu1[dnnResourceSrc] = resConv1[dnnResourceDst];
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_relu1_output, relu1, dnnResourceDst), err);
    CHECK_ERR(dnnAllocateBuffer_F64((void **)&resRelu1[dnnResourceDst], lt_relu1_output), err);

    /*** Convolution 2 section ***/
    if (!want_groups_conv)
    {
        CHECK_ERR(dnnConvolutionCreateForwardBias_F64(&conv2, attributes,
                                                      dnnAlgorithmConvolutionDirect, dimension, outputSize,
                                                      output2Size, filter2Size, convolutionStride, inputOffset,
                                                      dnnBorderZeros),
                  err);
    }
    else
    {
        CHECK_ERR(dnnGroupsConvolutionCreateForwardBias_F64(&conv2, attributes,
                                                            dnnAlgorithmConvolutionDirect, 1, dimension, outputSize,
                                                            output2Size, filter2Size, convolutionStride, inputOffset,
                                                            dnnBorderZeros),
                  err);
    }
    resConv2[dnnResourceSrc] = resRelu1[dnnResourceDst];
    resConv2[dnnResourceBias] = resConv1[dnnResourceBias];

    // Convolution describes what layout it expects
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv2_filt, conv2, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_conv2_output, conv2, dnnResourceDst), err);
    CHECK_ERR(init_conversion(&cv_user_to_conv2_filt, &resConv2[dnnResourceFilter], lt_conv2_filt, lt_user_filt2, user_c2_f), err);
    CHECK_ERR(dnnAllocateBuffer_F64((void **)&resConv2[dnnResourceDst], lt_conv2_output), err);

    /*** ReLU 2 section ***/
    CHECK_ERR(dnnReLUCreateForward_F64(&relu2, attributes, lt_conv2_output, 0.0f), err);
    resRelu2[dnnResourceSrc] = resConv2[dnnResourceDst];
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_relu2_output, relu2, dnnResourceDst), err);
    CHECK_ERR(dnnAllocateBuffer_F64((void **)&resRelu2[dnnResourceDst], lt_relu2_output), err);

    /*** Pooling section ***/
    lt_pool1_input = lt_relu2_output;
    CHECK_ERR(dnnPoolingCreateForward_F64(&pool1, attributes, dnnAlgorithmPoolingMax, lt_pool1_input, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err);
    resPool1[dnnResourceSrc] = resRelu2[dnnResourceDst];
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_pool1_output, pool1, dnnResourceDst), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_pool1_workspace, pool1, dnnResourceWorkspace), err);
    CHECK_ERR(dnnAllocateBuffer_F64(&resPool1[dnnResourceDst], lt_pool1_output), err);
    CHECK_ERR(dnnAllocateBuffer_F64(&resPool1[dnnResourceWorkspace], lt_pool1_workspace), err);

    CHECK_ERR(init_conversion(&cv_pool1_to_user_output, &user_o, lt_user_output, lt_pool1_output, resPool1[dnnResourceDst]), err);

    srand(1);

    for (int i = 0; i < inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]; i++)
        user_i[i] = (rand() % 200 - 100) / 100.;
    for (int i = 0; i < outputSize[2]; i++)
        user_c1_b[i] = (rand() % 200 - 100) / 100.;
    for (int i = 0; i < filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]; i++)
        user_c1_f[i] = (rand() % 200 - 100) / 100.;
    for (int i = 0; i < filter2Size[0] * filter2Size[1] * filter2Size[2] * filter2Size[3]; i++)
        user_c2_f[i] = (rand() % 200 - 100) / 100.;

    /*** Execution ***/
    if (cv_user_to_conv1_filt)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_filt, user_c1_f, resConv1[dnnResourceFilter]), err);
    if (cv_user_to_conv1_bias)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_bias, user_c1_b, resConv1[dnnResourceBias]), err);
    if (cv_user_to_conv2_filt)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv2_filt, user_c2_f, resConv2[dnnResourceFilter]), err);
    if (cv_user_to_conv1_input)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_input, user_i, resConv1[dnnResourceSrc]), err);

    double times[NB_TESTS];
    clock_t start, end;
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = clock();
        CHECK_ERR(dnnExecute_F64(conv1, (void *)resConv1), err);
        CHECK_ERR(dnnExecute_F64(relu1, (void *)resRelu1), err);
        CHECK_ERR(dnnExecute_F64(conv2, (void *)resConv2), err);
        CHECK_ERR(dnnExecute_F64(relu2, (void *)resRelu2), err);
        CHECK_ERR(dnnExecute_F64(pool1, (void *)resPool1), err);
        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }
    printf("\n\n\tConvolution time: %f.\n", median(NB_TESTS, times));

    if (cv_pool1_to_user_output)
        CHECK_ERR(dnnConversionExecute_F64(cv_pool1_to_user_output, resPool1[dnnResourceDst], user_o), err);

    FILE *f = fopen("mkl_result.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z)
            for (int y = 0; y < N - 2 * K; ++y)
                for (int x = 0; x < N - 2 * K; ++x)
                    fprintf(f, "%f\n", user_o[x + y * (N - 2 * K) + z * (N - 2 * K) * (N - 2 * K) + n * (N - 2 * K) * (N - 2 * K) * FOut]);

    fclose(f);

bail_out:

    dnnDelete_F64(conv1);
    dnnDelete_F64(relu1);
    dnnDelete_F64(conv2);
    dnnDelete_F64(pool1);
    dnnDelete_F64(cv_user_to_conv1_input);
    dnnDelete_F64(cv_user_to_conv1_filt);
    dnnDelete_F64(cv_user_to_conv1_bias);
    dnnDelete_F64(cv_user_to_conv2_filt);
    dnnDelete_F64(cv_pool1_to_user_output);

    dnnLayoutDelete_F64(lt_user_input);
    dnnLayoutDelete_F64(lt_user_filt);
    dnnLayoutDelete_F64(lt_user_filt2);
    dnnLayoutDelete_F64(lt_user_bias);
    dnnLayoutDelete_F64(lt_user_output);
    dnnLayoutDelete_F64(lt_conv1_input);
    dnnLayoutDelete_F64(lt_conv1_filt);
    dnnLayoutDelete_F64(lt_conv1_bias);
    dnnLayoutDelete_F64(lt_conv1_output);
    dnnLayoutDelete_F64(lt_relu1_output);
    dnnLayoutDelete_F64(lt_conv2_filt);
    dnnLayoutDelete_F64(lt_conv2_output);
    dnnLayoutDelete_F64(lt_pool1_output);
    dnnLayoutDelete_F64(lt_pool1_workspace);

    dnnPrimitiveAttributesDestroy_F64(attributes);
    if (resConv1[dnnResourceSrc] != (void *)user_i)
        dnnReleaseBuffer_F64(resConv1[dnnResourceSrc]);
    if (resConv1[dnnResourceFilter] != (void *)user_c1_f)
        dnnReleaseBuffer_F64(resConv1[dnnResourceFilter]);
    if (resConv1[dnnResourceBias] != (void *)user_c1_b)
        dnnReleaseBuffer_F64(resConv1[dnnResourceBias]);
    dnnReleaseBuffer_F64(resConv1[dnnResourceDst]);
    dnnReleaseBuffer_F64(resRelu1[dnnResourceDst]);
    if (resConv2[dnnResourceFilter] != (void *)user_c2_f)
        dnnReleaseBuffer_F64(resConv2[dnnResourceFilter]);
    dnnReleaseBuffer_F64(resConv2[dnnResourceDst]);
    dnnReleaseBuffer_F64(resPool1[dnnResourceDst]);
    dnnReleaseBuffer_F64(resPool1[dnnResourceWorkspace]);
    if ((void *)user_o != resPool1[dnnResourceDst])
        dnnReleaseBuffer_F64((void *)user_o);

    free(user_i);
    free(user_c1_f);
    free(user_c1_b);

    return err;
}

int main(int argc, char **argv)
{
    dnnError_t err;
    err = simple_net(0);
    if (err != E_SUCCESS)
    {
        printf("FAILED\n");
        return err;
    }

    printf("PASSED\n");
    return 0;
}
