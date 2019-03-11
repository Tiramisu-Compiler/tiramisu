/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

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
    size_t outputSize[dimension] = {(N - K_X + 2 * P_X) / S_X + 1, (N - K_Y + 2 * P_Y) / S_Y + 1, FIn, BATCH_SIZE};
    size_t outputStrides[dimension] = {1, (N - K_X + 2 * P_X) / S_X + 1,
                                       ((N - K_X + 2 * P_X) / S_X + 1) * ((N - K_Y + 2 * P_Y) / S_Y + 1),
                                       ((N - K_X + 2 * P_X) / S_X + 1) * ((N - K_Y + 2 * P_Y) / S_Y + 1) * FIn};

    size_t inputSize[dimension] = {N, N, FIn, BATCH_SIZE};
    size_t inputStrides[dimension] = {1, N, N * N, N * N * FIn};

    int inputOffset[dimension - 2] = {-P_X, -P_Y};

    dnnLayout_t lt_user_input = NULL,
                lt_user_output = NULL;
    dnnPrimitive_t cv_user_to_pool1_input = NULL;
    size_t kernelSize[2] = {K_X, K_Y};
    size_t kernelStride[2] = {S_X, S_Y};
    dnnLayout_t lt_pool1_input = NULL;
    dnnPrimitive_t pool1 = NULL;
    void *resPool1[dnnResourceNumber] = {0};
    dnnLayout_t lt_pool1_output = NULL,
                lt_pool1_workspace = NULL;
    dnnPrimitive_t cv_pool1_to_user_output = NULL;
    dnnPrimitiveAttributes_t attributes = NULL;

    double *user_i = NULL,
           *user_o = NULL;

    /*** data allocation ***/
    user_i = (double *)malloc(sizeof(double) * (inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]));

    if (user_i == NULL)
    {
        err = E_MEMORY_ERROR;
        goto bail_out;
    }

    /*** User's data description ***/
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_input, dimension, inputSize, inputStrides), err);
    CHECK_ERR(dnnLayoutCreate_F64(&lt_user_output, dimension, outputSize, outputStrides), err);

    /* Initialize attributes */
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    /*** Pooling section ***/
    CHECK_ERR(dnnPoolingCreateForward_F64(&pool1, attributes, dnnAlgorithmPoolingMax,
                                          lt_user_input, kernelSize, kernelStride, inputOffset, dnnBorderZeros),
              err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_pool1_input, pool1, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_pool1_output, pool1, dnnResourceDst), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_pool1_workspace, pool1, dnnResourceWorkspace), err);
    CHECK_ERR(init_conversion(&cv_user_to_pool1_input, &resPool1[dnnResourceSrc], lt_pool1_input, lt_user_input, user_i), err);

    CHECK_ERR(dnnAllocateBuffer_F64(&resPool1[dnnResourceDst], lt_pool1_output), err);
    CHECK_ERR(dnnAllocateBuffer_F64(&resPool1[dnnResourceWorkspace], lt_pool1_workspace), err);

    CHECK_ERR(init_conversion(&cv_pool1_to_user_output, &user_o, lt_user_output, lt_pool1_output, resPool1[dnnResourceDst]), err);

    srand(1);
    for (int i = 0; i < inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]; i++)
        user_i[i] = rand() % 10;

    /*** Execution ***/

    if (cv_user_to_pool1_input)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_pool1_input, user_i, resPool1[dnnResourceSrc]), err);

    double times[NB_TESTS];
    clock_t start, end;
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = clock();
        CHECK_ERR(dnnExecute_F64(pool1, (void *)resPool1), err);
        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }
    printf("Pool time: %f.\n", median(NB_TESTS, times));

    if (cv_pool1_to_user_output)
        CHECK_ERR(dnnConversionExecute_F64(cv_pool1_to_user_output, resPool1[dnnResourceDst], user_o), err);
    FILE *f = fopen("mkl_result.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < outputSize[1]; ++y)
                for (int x = 0; x < outputSize[0]; ++x)
                    fprintf(f, "%.0f", user_o[x + y * outputSize[0] + z * outputSize[0] * outputSize[1] + n * outputSize[0] * outputSize[1] * FIn]);

    fclose(f);

bail_out:

    dnnDelete_F64(pool1);
    dnnDelete_F64(cv_user_to_pool1_input);
    dnnDelete_F64(cv_pool1_to_user_output);
    dnnLayoutDelete_F64(lt_user_input);
    dnnLayoutDelete_F64(lt_user_output);
    dnnLayoutDelete_F64(lt_pool1_input);
    dnnLayoutDelete_F64(lt_pool1_output);
    dnnLayoutDelete_F64(lt_pool1_workspace);
    dnnPrimitiveAttributesDestroy_F64(attributes);
    dnnReleaseBuffer_F64(resPool1[dnnResourceDst]);
    dnnReleaseBuffer_F64(resPool1[dnnResourceWorkspace]);
    if ((void *)user_o != resPool1[dnnResourceDst])
        dnnReleaseBuffer_F64((void *)user_o);

    free(user_i);

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
