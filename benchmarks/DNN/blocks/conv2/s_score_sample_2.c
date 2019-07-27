#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "configure.h"

#include "mkl_dnn.h"

// MKL-DNN default format is NCHW according to
// https://www.tensorflow.org/performance/performance_guide

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
        goto bail_out; \
    } \
} while(0)

#define dimension (4)

static dnnError_t init_conversion(dnnPrimitive_t *cv, float **ptr_out,
                                 dnnLayout_t lt_pr, dnnLayout_t lt_us, float *ptr_us)
{
    dnnError_t err;
    *ptr_out = NULL;
    if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
        CHECK_ERR( dnnConversionCreate_F32(cv, lt_us, lt_pr), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err );
    } else {
        *ptr_out = ptr_us;
    }
    return E_SUCCESS;

bail_out:
    if (*ptr_out) dnnReleaseBuffer_F32(*ptr_out);
    return err;
}

static dnnError_t simple_net(int want_groups_conv) {
    dnnError_t err;

    size_t outputSize[dimension] = { (N-4), (N-4), FOut, BATCH_SIZE };
    size_t outputStrides[dimension] = { 1, (N-4), (N-4) * (N-4), (N-4) * (N-4) * FOut };

    size_t inputSize[dimension] = { N, N, FIn, BATCH_SIZE };
    size_t inputStrides[dimension] = { 1, N, (N) * (N), (N) * (N) * FIn };

    size_t filterSize[dimension] = { K, K, FIn, FOut };
    size_t filterStrides[dimension] = { 1, K, K * K, K * K * FIn };

    size_t convolutionStride[dimension - 2] = { 1, 1 };
    int inputOffset[dimension - 2 ] = { 0, 0 };

    size_t biasSize[1] = { outputSize[2] };
    size_t biasStrides[1] = { 1 };

    dnnLayout_t lt_user_input = NULL,
                lt_user_filt = NULL,
                lt_user_bias = NULL,
                lt_user_filt_2 = NULL,
                lt_user_bias_2 = NULL,
                lt_user_output = NULL;
    dnnLayout_t lt_conv1_input = NULL,
                lt_conv1_filt = NULL,
                lt_conv1_bias = NULL,
                lt_conv1_output = NULL,
                lt_conv2_filt = NULL,
                lt_conv2_bias = NULL,
                lt_conv2_output = NULL;
    float* resConv1[dnnResourceNumber] = {0};
    float* resConv2[dnnResourceNumber] = {0};
    dnnPrimitive_t cv_user_to_conv1_input = NULL,
                   cv_user_to_conv1_filt = NULL,
                   cv_user_to_conv1_bias = NULL,
                   cv_user_to_conv2_filt = NULL,
                   cv_user_to_conv2_bias = NULL;
    dnnPrimitive_t conv1 = NULL;
    dnnPrimitive_t conv2 = NULL;
    dnnPrimitive_t cv_conv2_to_user_output = NULL;
    dnnPrimitiveAttributes_t attributes = NULL;

    float *user_i = (float*)malloc(sizeof(float)*inputSize[0]*inputSize[1]*inputSize[2]*inputSize[3]);
    float *user_c1_f = (float*)malloc(sizeof(float)*(filterSize[0]*filterSize[1]*filterSize[2]*filterSize[3]));
    float *user_c1_b = (float*)malloc(sizeof(float)*(inputSize[2]));
    float *user_c1_o = (float*)malloc(sizeof(float)*outputSize[0]*outputSize[1]*outputSize[2]*outputSize[3]);
    float *user_c2_f = (float*)malloc(sizeof(float)*(filterSize[0]*filterSize[1]*filterSize[2]*filterSize[3]));
    float *user_c2_b = (float*)malloc(sizeof(float)*(inputSize[2]));
    float *user_c2_o = (float*)malloc(sizeof(float)*outputSize[0]*outputSize[1]*outputSize[2]*outputSize[3]);


    /*** User's data description ***/
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filt, dimension, filterSize, filterStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filt_2, dimension, filterSize, filterStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias_2, 1, biasSize, biasStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );


    /* Initialize attributes */
    CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );

    /*** convolution section ***/
    CHECK_ERR( dnnConvolutionCreateForwardBias_F32(&conv1, attributes,
                    dnnAlgorithmConvolutionDirect, dimension, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros),
               err );
    CHECK_ERR( dnnConvolutionCreateForwardBias_F32(&conv2, attributes,
                    dnnAlgorithmConvolutionDirect, dimension, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros),
               err );


    // Convolution describes what layout it expects
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv1, dnnResourceSrc) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv1, dnnResourceFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, conv1, dnnResourceBias) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output, conv1, dnnResourceDst) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv2_filt, conv2, dnnResourceFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv2_bias, conv2, dnnResourceBias) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_conv2_output, conv2, dnnResourceDst) , err );

    CHECK_ERR( init_conversion(&cv_user_to_conv1_input, &resConv1[dnnResourceSrc], lt_conv1_input, lt_user_input, user_i) , err );
    CHECK_ERR( init_conversion(&cv_user_to_conv1_filt, &resConv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt, user_c1_f), err );
    CHECK_ERR( init_conversion(&cv_user_to_conv1_bias, &resConv1[dnnResourceBias], lt_conv1_bias, lt_user_bias, user_c1_b), err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDst], lt_conv1_output), err );
    CHECK_ERR( init_conversion(&cv_user_to_conv2_filt, &resConv2[dnnResourceFilter], lt_conv2_filt, lt_user_filt_2, user_c2_f), err );
    CHECK_ERR( init_conversion(&cv_user_to_conv2_bias, &resConv2[dnnResourceBias], lt_conv2_bias, lt_user_bias_2, user_c2_b), err );
    CHECK_ERR( dnnAllocateBuffer_F32((void**)&resConv2[dnnResourceDst], lt_conv2_output), err );

    CHECK_ERR( dnnAllocateBuffer_F32(&resConv1[dnnResourceDst], lt_conv1_output) , err );
    CHECK_ERR( dnnAllocateBuffer_F32(&resConv2[dnnResourceDst], lt_conv2_output) , err );
    CHECK_ERR( init_conversion(&cv_conv2_to_user_output, &user_c2_o, lt_user_output, lt_conv2_output, resConv2[dnnResourceDst]), err );

    for (int i=0; i<inputSize[0]*inputSize[1]*inputSize[2]*inputSize[3]; i++) user_i[i] = 1;
    for (int i=0; i<filterSize[0]*filterSize[1]*filterSize[2]*filterSize[3]; i++) user_c1_f[i] = 1;
    for (int i=0; i<inputSize[2]; i++) user_c1_b[i] = 1;
    for (int i=0; i<outputSize[0]*outputSize[1]*outputSize[2]*outputSize[3]; i++) user_c1_o[i] = 9;
    for (int i=0; i<filterSize[0]*filterSize[1]*filterSize[2]*filterSize[3]; i++) user_c2_f[i] = 1;
    for (int i=0; i<inputSize[2]; i++) user_c2_b[i] = 1;
    for (int i=0; i<outputSize[0]*outputSize[1]*outputSize[2]*outputSize[3]; i++) user_c2_o[i] = 9;


    /** Conv 2 **/
    resConv2[dnnResourceSrc] = resConv1[dnnResourceDst];


    /*** Execution ***/
    if (cv_user_to_conv1_filt) CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_filt, user_c1_f, resConv1[dnnResourceFilter]), err );
    if (cv_user_to_conv1_bias) CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_bias, user_c1_b, resConv1[dnnResourceBias]), err );
    if (cv_user_to_conv1_input) CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv1_input, user_i, resConv1[dnnResourceSrc]), err );
    if (cv_user_to_conv2_filt) CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv2_filt, user_c2_f, resConv2[dnnResourceFilter]), err );
    if (cv_user_to_conv2_bias) CHECK_ERR( dnnConversionExecute_F32(cv_user_to_conv2_bias, user_c2_b, resConv2[dnnResourceBias]), err );


    double start, end;
    start = rtclock();
    CHECK_ERR( dnnExecute_F32(conv1, (void*)resConv1), err );
    CHECK_ERR( dnnExecute_F32(conv2, (void*)resConv2), err );
    end = rtclock();
    double time_taken = (end - start)*1000;
    printf("Convolution time: %f.\n",  time_taken);

    if (cv_conv2_to_user_output) CHECK_ERR( dnnConversionExecute_F32(cv_conv2_to_user_output, resConv2[dnnResourceDst], user_c2_o), err );

    for (int i=0; i<(PRINT_ONLY_10?10:outputSize[0]*outputSize[1]*outputSize[2]*outputSize[3]); i++)
	printf("%f, ", user_c2_o[i]);

bail_out:

    dnnDelete_F32(conv1);
    dnnDelete_F32(cv_user_to_conv1_input);
    dnnDelete_F32(cv_user_to_conv1_filt);
    dnnDelete_F32(cv_user_to_conv1_bias);

    dnnLayoutDelete_F32(lt_user_input);
    dnnLayoutDelete_F32(lt_user_filt);
    dnnLayoutDelete_F32(lt_user_bias);
    dnnLayoutDelete_F32(lt_user_filt_2);
    dnnLayoutDelete_F32(lt_user_bias_2);
    dnnLayoutDelete_F32(lt_user_output);
    dnnLayoutDelete_F32(lt_conv1_input);
    dnnLayoutDelete_F32(lt_conv1_filt);
    dnnLayoutDelete_F32(lt_conv1_bias);
    dnnLayoutDelete_F32(lt_conv1_output);

    dnnPrimitiveAttributesDestroy_F32(attributes);
    if (resConv1[dnnResourceSrc] != (void*)user_i) dnnReleaseBuffer_F32(resConv1[dnnResourceSrc]);
    if (resConv1[dnnResourceFilter] != (void*)user_c1_f) dnnReleaseBuffer_F32(resConv1[dnnResourceFilter]);
    if (resConv1[dnnResourceBias] != (void*)user_c1_b) dnnReleaseBuffer_F32(resConv1[dnnResourceBias]);
    dnnReleaseBuffer_F32(resConv1[dnnResourceDst]);

    free(user_i);
    free(user_c1_f);
    free(user_c1_b);

    return err;
}

int main(int argc, char **argv) {
    dnnError_t err;
    err = simple_net(0);
    if (err != E_SUCCESS) { printf("FAILED\n"); return err; }

    printf("PASSED\n");
    return 0;
}
