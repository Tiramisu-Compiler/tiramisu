#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "configure.h"

#include "mkl_dnn.h"

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

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

static dnnError_t simple_net(int want_groups_conv)
{
    dnnError_t err;

    int nb_sizes;

    if (RUN_DIFFERENT_SIZES)
	nb_sizes = 27;
    else
	nb_sizes = 1;

    int sizes[nb_sizes][4];

    if (RUN_DIFFERENT_SIZES)
    {
	sizes[0][0] = 32;
	sizes[0][1] = 8;
	sizes[0][2] = 16;
	sizes[0][3] = 16;

	sizes[1][0] = 64;
	sizes[1][1] = 32;
	sizes[1][2] = 16;
	sizes[1][3] = 16;

	sizes[2][0] = 512;
	sizes[2][1] = 100;
	sizes[2][2] = 16;
	sizes[2][3] = 16;

	sizes[3][0] = 224;
	sizes[3][1] = 8;
	sizes[3][2] = 3;
	sizes[3][3] = 64;

	sizes[4][0] = 224;
	sizes[4][1] = 32;
	sizes[4][2] = 3;
	sizes[4][3] = 64;

	sizes[5][0] = 224;
	sizes[5][1] = 100;
	sizes[5][2] = 3;
	sizes[5][3] = 64;

	sizes[6][0] = 56;
	sizes[6][1] = 8;
	sizes[6][2] = 64;
	sizes[6][3] = 64;

	sizes[7][0] = 56;
	sizes[7][1] = 32;
	sizes[7][2] = 64;
	sizes[7][3] = 64;

	sizes[8][0] = 56;
	sizes[8][1] = 100;
	sizes[8][2] = 64;
	sizes[8][3] = 64;

	sizes[9][0] = 56;
	sizes[9][1] = 8;
	sizes[9][2] = 64;
	sizes[9][3] = 128;

	sizes[10][0] = 56;
	sizes[10][1] = 32;
	sizes[10][2] = 64;
	sizes[10][3] = 128;

	sizes[11][0] = 56;
	sizes[11][1] = 100;
	sizes[11][2] = 64;
	sizes[11][3] = 128;

	sizes[12][0] = 28;
	sizes[12][1] = 8;
	sizes[12][2] = 128;
	sizes[12][3] = 128;

	sizes[13][0] = 28;
	sizes[13][1] = 32;
	sizes[13][2] = 128;
	sizes[13][3] = 128;

	sizes[14][0] = 28;
	sizes[14][1] = 100;
	sizes[14][2] = 128;
	sizes[14][3] = 128;

	sizes[15][0] = 28;
	sizes[15][1] = 8;
	sizes[15][2] = 100;
	sizes[15][3] = 256;

	sizes[16][0] = 28;
	sizes[16][1] = 32;
	sizes[16][2] = 100;
	sizes[16][3] = 256;

	sizes[17][0] = 28;
	sizes[17][1] = 100;
	sizes[17][2] = 100;
	sizes[17][3] = 256;

	sizes[18][0] = 14;
	sizes[18][1] = 8;
	sizes[18][2] = 256;
	sizes[18][3] = 256;

	sizes[19][0] = 14;
	sizes[19][1] = 32;
	sizes[19][2] = 256;
	sizes[19][3] = 256;

	sizes[20][0] = 14;
	sizes[20][1] = 100;
	sizes[20][2] = 256;
	sizes[20][3] = 256;

	sizes[21][0] = 14;
	sizes[21][1] = 8;
	sizes[21][2] = 310;
	sizes[21][3] = 512;

	sizes[22][0] = 14;
	sizes[22][1] = 32;
	sizes[22][2] = 310;
	sizes[22][3] = 512;

	sizes[23][0] = 14;
	sizes[23][1] = 100;
	sizes[23][2] = 310;
	sizes[23][3] = 512;

	sizes[24][0] = 7;
	sizes[24][1] = 8;
	sizes[24][2] = 512;
	sizes[24][3] = 512;

	sizes[25][0] = 7;
	sizes[25][1] = 32;
	sizes[25][2] = 512;
	sizes[25][3] = 512;

	sizes[26][0] = 7;
	sizes[26][1] = 100;
	sizes[26][2] = 512;
	sizes[26][3] = 512;
    }
    else
    {
	sizes[0][0] = 512;
	sizes[0][1] = 100;
	sizes[0][2] = 16;
	sizes[0][3] = 16;
    }

    FILE *f = fopen("mkl_result.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int j = 0; j < nb_sizes; j++)
    {
        int N = sizes[j][0];
        int BATCH_SIZE = sizes[j][1];
        int FIn = sizes[j][2];
        int FOut = sizes[j][3];

        size_t outputSize[dimension] = {(N - 4), (N - 4), FOut, BATCH_SIZE};
        size_t outputStrides[dimension] = {1, (N - 4), (N - 4) * (N - 4), (N - 4) * (N - 4) * FOut};

        size_t inputSize[dimension] = {N, N, FIn, BATCH_SIZE};
        size_t inputStrides[dimension] = {1, N, (N) * (N), (N) * (N)*FIn};

        size_t filterSize[dimension] = {K, K, FIn, FOut};
        size_t filterStrides[dimension] = {1, K, K * K, K * K * FIn};

        size_t convolutionStride[dimension - 2] = {1, 1};
        int inputOffset[dimension - 2] = {0, 0};

        size_t biasSize[1] = {outputSize[2]};
        size_t biasStrides[1] = {1};

        dnnLayout_t lt_user_input = NULL,
                    lt_user_filt = NULL,
                    lt_user_bias = NULL,
                    lt_user_output = NULL;
        dnnLayout_t lt_conv1_input = NULL,
                    lt_conv1_filt = NULL,
                    lt_conv1_bias = NULL,
                    lt_conv1_output = NULL;
        float *resConv1[dnnResourceNumber] = {0};
        dnnPrimitive_t cv_user_to_conv1_input = NULL,
                       cv_user_to_conv1_filt = NULL,
                       cv_user_to_conv1_bias = NULL;
        dnnPrimitive_t conv1 = NULL;
        dnnPrimitive_t cv_conv1_to_user_output = NULL;
        dnnPrimitiveAttributes_t attributes = NULL;

        float *user_i = (float *)malloc(sizeof(float) * inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]);
        float *user_c1_f = (float *)malloc(sizeof(float) * (filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]));
        float *user_c1_b = (float *)malloc(sizeof(float) * (inputSize[2]));
        float *user_o = (float *)malloc(sizeof(float) * outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3]);

        /*** User's data description ***/
        CHECK_ERR(dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err);
        CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt, dimension, filterSize, filterStrides), err);
        CHECK_ERR(dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides), err);
        CHECK_ERR(dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err);

        /* Initialize attributes */
        CHECK_ERR(dnnPrimitiveAttributesCreate_F32(&attributes), err);

        /*** convolution section ***/
        CHECK_ERR(dnnConvolutionCreateForwardBias_F32(&conv1, attributes,
                                                      dnnAlgorithmConvolutionDirect, dimension, inputSize,
                                                      outputSize, filterSize, convolutionStride, inputOffset,
                                                      dnnBorderZeros),
                  err);

        // Convolution describes what layout it expects
        CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv1, dnnResourceSrc), err);
        CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv1, dnnResourceFilter), err);
        CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, conv1, dnnResourceBias), err);
        CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output, conv1, dnnResourceDst), err);

        CHECK_ERR(init_conversion(&cv_user_to_conv1_input, &resConv1[dnnResourceSrc], lt_conv1_input, lt_user_input, user_i), err);
        CHECK_ERR(init_conversion(&cv_user_to_conv1_filt, &resConv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt, user_c1_f), err);
        CHECK_ERR(init_conversion(&cv_user_to_conv1_bias, &resConv1[dnnResourceBias], lt_conv1_bias, lt_user_bias, user_c1_b), err);
        CHECK_ERR(dnnAllocateBuffer_F32((void **)&resConv1[dnnResourceDst], lt_conv1_output), err);

        CHECK_ERR(dnnAllocateBuffer_F32(&resConv1[dnnResourceDst], lt_conv1_output), err);
        CHECK_ERR(init_conversion(&cv_conv1_to_user_output, &user_o, lt_user_output, lt_conv1_output, resConv1[dnnResourceDst]), err);

        for (int i = 0; i < inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]; i++)
            user_i[i] = 1;
        for (int i = 0; i < filterSize[0] * filterSize[1] * filterSize[2] * filterSize[3]; i++)
            user_c1_f[i] = 1;
        for (int i = 0; i < inputSize[2]; i++)
            user_c1_b[i] = 1;
        for (int i = 0; i < outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3]; i++)
            user_o[i] = 9;

        /*** Execution ***/
        if (cv_user_to_conv1_filt)
            CHECK_ERR(dnnConversionExecute_F32(cv_user_to_conv1_filt, user_c1_f, resConv1[dnnResourceFilter]), err);
        if (cv_user_to_conv1_bias)
            CHECK_ERR(dnnConversionExecute_F32(cv_user_to_conv1_bias, user_c1_b, resConv1[dnnResourceBias]), err);
        if (cv_user_to_conv1_input)
            CHECK_ERR(dnnConversionExecute_F32(cv_user_to_conv1_input, user_i, resConv1[dnnResourceSrc]), err);

        double times[NB_TESTS];
        clock_t start, end;
        for (int i = 0; i < NB_TESTS; i++)
        {
            start = clock();
            CHECK_ERR(dnnExecute_F32(conv1, (void *)resConv1), err);
            end = clock();
            double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
            times[i] = time_taken;
        }
        printf("Convolution time: %f.\n", median(NB_TESTS, times));
        fprintf(f, "%f\n", median(NB_TESTS, times));

        if (cv_conv1_to_user_output)
            CHECK_ERR(dnnConversionExecute_F32(cv_conv1_to_user_output, resConv1[dnnResourceDst], user_o), err);

        for (int i = 0; i < (PRINT_ONLY_10 ? 10 : outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3]); i++)
            printf("%f, ", user_o[i]);

    bail_out:

        dnnDelete_F32(conv1);
        dnnDelete_F32(cv_user_to_conv1_input);
        dnnDelete_F32(cv_user_to_conv1_filt);
        dnnDelete_F32(cv_user_to_conv1_bias);

        dnnLayoutDelete_F32(lt_user_input);
        dnnLayoutDelete_F32(lt_user_filt);
        dnnLayoutDelete_F32(lt_user_bias);
        dnnLayoutDelete_F32(lt_user_output);
        dnnLayoutDelete_F32(lt_conv1_input);
        dnnLayoutDelete_F32(lt_conv1_filt);
        dnnLayoutDelete_F32(lt_conv1_bias);
        dnnLayoutDelete_F32(lt_conv1_output);

        dnnPrimitiveAttributesDestroy_F32(attributes);
        if (resConv1[dnnResourceSrc] != (void *)user_i)
            dnnReleaseBuffer_F32(resConv1[dnnResourceSrc]);
        if (resConv1[dnnResourceFilter] != (void *)user_c1_f)
            dnnReleaseBuffer_F32(resConv1[dnnResourceFilter]);
        if (resConv1[dnnResourceBias] != (void *)user_c1_b)
            dnnReleaseBuffer_F32(resConv1[dnnResourceBias]);
        dnnReleaseBuffer_F32(resConv1[dnnResourceDst]);

        free(user_i);
        free(user_c1_f);
        free(user_c1_b);
    }
    fclose(f);
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
