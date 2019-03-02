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
    size_t outputSize[dimension] = {(N - KERNEL + 2 * PADDING) / STRIDES + 1, (N - KERNEL + 2 * PADDING) / STRIDES + 1, FOut, BATCH_SIZE};
    size_t outputStrides[dimension] = {1, (N - KERNEL + 2 * PADDING) / STRIDES + 1, ((N - KERNEL + 2 * PADDING) / STRIDES + 1) * ((N - KERNEL + 2 * PADDING) / STRIDES + 1), ((N - KERNEL + 2 * PADDING) / STRIDES + 1) * ((N - KERNEL + 2 * PADDING) / STRIDES + 1) * FOut};

    size_t inputSize[dimension] = {N, N, FIn, BATCH_SIZE};
    size_t inputStrides[dimension] = {1, N, N * N, N * N * FIn};

    size_t filterSize[dimension] = {KERNEL, KERNEL, FIn, FOut};
    size_t filterStrides[dimension] = {1, KERNEL, KERNEL * KERNEL, KERNEL * KERNEL * FIn};

    size_t convolutionStride[dimension - 2] = {STRIDES, STRIDES};
    int inputOffset[dimension - 2] = {-PADDING, -PADDING};

    size_t biasSize[1] = {outputSize[2]};
    size_t biasStrides[1] = {outputStrides[2]};

    dnnLayout_t lt_user_input = NULL,
                lt_user_filt = NULL,
                lt_user_bias = NULL,
                lt_user_output = NULL;
    dnnPrimitive_t conv1 = NULL;
    dnnLayout_t lt_conv1_input = NULL,
                lt_conv1_filt = NULL,
                lt_conv1_bias = NULL,
                lt_conv1_output = NULL;
    double *resConv1[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_user_to_conv1_input = NULL,
                   cv_user_to_conv1_filt = NULL,
                   cv_user_to_conv1_bias = NULL;
    dnnPrimitive_t cv_conv1_to_user_output = NULL;
    dnnPrimitiveAttributes_t attributes = NULL;

    double *user_i = NULL,
           *user_c1_f = NULL,
           *user_c1_b = NULL,
           *user_o = NULL;

    /*** data allocation ***/
    user_i = (double *)malloc(sizeof(double) * (inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]));
    user_c1_f = (double *)malloc(sizeof(double) * (filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]));
    user_c1_b = (double *)malloc(sizeof(double) * (8 * 1024 * 1024));
    if (user_i == NULL || user_c1_f == NULL || user_c1_b == NULL)
    {
        err = E_MEMORY_ERROR;
        goto bail_out;
    }

    /*** User's data description ***/
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_input, dimension, inputSize, inputStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_filt, dimension, filterSize, filterStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_bias, 1, biasSize, biasStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_output, dimension, outputSize, outputStrides), err);

    /* Initialize attributes */
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    /*** convolution section ***/
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
    CHECK_ERR(init_conversion(&cv_conv1_to_user_output, &user_o, lt_user_output, lt_conv1_output, resConv1[dnnResourceDst]), err);

    srand(1);

    for (int i = 0; i < inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]; i++)
        user_i[i] = rand() % 10;
    for (int i = 0; i < filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]; i++)
        user_c1_f[i] = 1;
    for (int i = 0; i < outputSize[2]; i++)
        user_c1_b[i] = 0;

    /*** Execution ***/
    if (cv_user_to_conv1_filt)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_filt, user_c1_f, resConv1[dnnResourceFilter]), err);
    if (cv_user_to_conv1_bias)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_bias, user_c1_b, resConv1[dnnResourceBias]), err);

    if (cv_user_to_conv1_input)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_conv1_input, user_i, resConv1[dnnResourceSrc]), err);
    double times[NB_TESTS];
    clock_t start, end;
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = clock();
        CHECK_ERR(dnnExecute_F64(conv1, (void *)resConv1), err);
        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }
    printf("Convolution time: %f.\n", median(NB_TESTS, times));

    if (cv_conv1_to_user_output)
        CHECK_ERR(dnnConversionExecute_F64(cv_conv1_to_user_output, resConv1[dnnResourceDst], user_o), err); // Shift pointers

    FILE *f = fopen("mkl_result.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FOut; ++z)
            for (int y = 0; y < outputSize[1]; ++y)
                for (int x = 0; x < outputSize[0]; ++x)
                    fprintf(f, "%.0f\n", user_o[x + y * outputSize[0] + z * outputSize[0] * outputSize[1] + n * outputSize[0] * outputSize[1] * FOut]);

    fclose(f);
bail_out:

    dnnDelete_F64(conv1);

    dnnDelete_F64(cv_user_to_conv1_input);
    dnnDelete_F64(cv_user_to_conv1_filt);
    dnnDelete_F64(cv_user_to_conv1_bias);
    dnnDelete_F64(cv_conv1_to_user_output);

    dnnLayoutDelete_F64(lt_user_input);
    dnnLayoutDelete_F64(lt_user_filt);
    dnnLayoutDelete_F64(lt_user_bias);
    dnnLayoutDelete_F64(lt_user_output);
    dnnLayoutDelete_F64(lt_conv1_input);
    dnnLayoutDelete_F64(lt_conv1_filt);
    dnnLayoutDelete_F64(lt_conv1_bias);
    dnnLayoutDelete_F64(lt_conv1_output);

    dnnPrimitiveAttributesDestroy_F64(attributes);
    if (resConv1[dnnResourceSrc] != (void *)user_i)
        dnnReleaseBuffer_F64(resConv1[dnnResourceSrc]);
    if (resConv1[dnnResourceFilter] != (void *)user_c1_f)
        dnnReleaseBuffer_F64(resConv1[dnnResourceFilter]);
    if (resConv1[dnnResourceBias] != (void *)user_c1_b)
        dnnReleaseBuffer_F64(resConv1[dnnResourceBias]);
    dnnReleaseBuffer_F64(resConv1[dnnResourceDst]);

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
    err = simple_net(1);
    if (err != E_SUCCESS)
    {
        printf("FAILED\n");
        return err;
    }

    printf("PASSED\n");
    return 0;
}
