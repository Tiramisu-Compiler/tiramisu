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

static dnnError_t simple_net()
{
    dnnError_t err;

    size_t outputSize[dimension] = {N, N, FIn, BATCH_SIZE};
    size_t outputStrides[dimension] = {1, N, N * N, N * N * FIn};

    size_t inputSize[dimension] = {N, N, FIn, BATCH_SIZE};
    size_t inputStrides[dimension] = {1, N, N * N, N * N * FIn};

    dnnLayout_t lt_user_input = NULL,
                lt_user_output = NULL;
    dnnLayout_t lt_relu1_input = NULL,
                lt_relu1_output = NULL;
    dnnPrimitive_t cv_user_to_relu1_input = NULL,
                   cv_relu1_to_user_output = NULL;
    dnnPrimitive_t relu1 = NULL;
    double *resRelu1[dnnResourceNumber] = {0};
    dnnPrimitiveAttributes_t attributes = NULL;

    double *user_i = NULL,
           *user_o = NULL;

    /*** data allocation ***/
    user_i = (double *)malloc(sizeof(double) * (N * N * FIn * BATCH_SIZE));
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

    /*** ReLU section ***/
    CHECK_ERR(dnnReLUCreateForward_F64(&relu1, attributes, lt_user_input, -0.01), err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_relu1_input, relu1, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F64(&lt_relu1_output, relu1, dnnResourceDst), err);
    CHECK_ERR(init_conversion(&cv_user_to_relu1_input, &resRelu1[dnnResourceSrc], lt_relu1_input, lt_user_input, user_i), err);
    CHECK_ERR(dnnAllocateBuffer_F64((void **)&resRelu1[dnnResourceDst], lt_relu1_output), err);
    CHECK_ERR(init_conversion(&cv_relu1_to_user_output, &user_o, lt_user_output, lt_relu1_output, resRelu1[dnnResourceDst]), err);

    srand(1);

    for (int i = 0; i < inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3]; i++)
        user_i[i] = rand() % 10 - 5;

    /*** Execution ***/
    if (cv_user_to_relu1_input)
        CHECK_ERR(dnnConversionExecute_F64(cv_user_to_relu1_input, user_i, resRelu1[dnnResourceSrc]), err);

    double times[NB_TESTS];
    clock_t start, end;
    for (int i = 0; i < NB_TESTS; i++)
    {
        start = clock();
        CHECK_ERR(dnnExecute_F64(relu1, (void *)resRelu1), err);
        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }
    printf("Relu time: %f.\n", median(NB_TESTS, times));

    if (cv_relu1_to_user_output)
        CHECK_ERR(dnnConversionExecute_F64(cv_relu1_to_user_output, resRelu1[dnnResourceDst], user_o), err); // Shift pointers

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
                    fprintf(f, "%.0f\n", user_o[x + y * outputSize[0] + z * outputSize[0] * outputSize[1] + n * outputSize[0] * outputSize[1] * FIn]);

    fclose(f);

bail_out:

    dnnDelete_F64(relu1);
    dnnDelete_F64(cv_user_to_relu1_input);
    dnnLayoutDelete_F64(lt_user_input);
    dnnLayoutDelete_F64(lt_user_output);
    dnnLayoutDelete_F64(lt_relu1_input);
    dnnLayoutDelete_F64(lt_relu1_output);

    dnnPrimitiveAttributesDestroy_F64(attributes);
    if (resRelu1[dnnResourceSrc] != (void *)user_i)
        dnnReleaseBuffer_F64(resRelu1[dnnResourceSrc]);
    dnnReleaseBuffer_F64(resRelu1[dnnResourceDst]);

    free(user_i);

    return err;
}

int main(int argc, char **argv)
{
    dnnError_t err;
    err = simple_net();
    if (err != E_SUCCESS)
    {
        printf("FAILED\n");
        return err;
    }

    printf("PASSED\n");
    return 0;
}
