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
    size_t input_size[] = {N, N, FIn, BATCH_SIZE};
    size_t input_strides[] = {1, N, N*N, N*N*FIn};

    float bn_scale_shift[2*FIn];
    float bn_workspace[2*FIn];

    for (int z = 0; z < FIn; ++z) {
        bn_scale_shift[z] = 1.f;
        bn_scale_shift[z + FIn] = 0.f;
    }

    // Create batch normalization primitive
    dnnLayout_t input_layout;
    CHECK_ERR(dnnLayoutCreate_F32(&input_layout, TENSOR_DIMENSION, input_size, input_strides), err);

    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F32(&attributes), err);

    dnnPrimitive_t bn_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F32(&bn_primitive, attributes, input_layout, EPSILON), err);

    // Allocate buffers
    float* input_buf = malloc(sizeof(float) * N * N * FIn * BATCH_SIZE);
    float* output_buf = malloc(sizeof(float) * N * N * FIn * BATCH_SIZE);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*FIn] = (float)(rand() % 100);

    // Execute the primitive
    float* resources[dnnResourceNumber] = {0};
    double times[NB_TESTS];
    double start, end;

    for (int i = 0; i < NB_TESTS; ++i) {
        start = rtclock();

        resources[dnnResourceSrc] = input_buf;
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = output_buf;
        CHECK_ERR(dnnExecute_F32(bn_primitive, (void**)resources), err);

        end = rtclock();
        times[i] = (end - start) * 1000;
    }

    printf("\n\n\tMKL BN duration: %f ms.\n", median(NB_TESTS, times));

    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 1;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < FIn; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output_buf[x + y*N + z*N*N + n*N*N*FIn]);

    fclose(f);
    
    free(output_buf);
    free(input_buf);
    dnnDelete_F32(bn_primitive);
    dnnLayoutDelete_F32(input_layout);
    
    return 0;
}
