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

dnnError_t init_conversion(dnnPrimitive_t *cv, double **ptr_out,
                           dnnLayout_t lt_pr, dnnLayout_t lt_us, double *ptr_us)
{
    dnnError_t err;
    *ptr_out = NULL;

    if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
        CHECK_ERR(dnnConversionCreate_F32(cv, lt_us, lt_pr), err);
        CHECK_ERR(dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err);
    }

    else
        *ptr_out = ptr_us;

    return E_SUCCESS;
}

int main()
{
    srand(1);
    dnnError_t err;

    // Define some parameters
    size_t input_size[] = {N, N, 4*GR, BATCH_SIZE};
    size_t input_strides[] = {1, N, N*N, N*N*4*GR};
    size_t output_size[] = {N, N, GR, BATCH_SIZE};

    double bn_scale_shift[2*4*GR];
    double bn_workspace[2*4*GR];

    double conv_filter_param[GR][4*GR][K_Y][K_X];
    double conv_bias_param[GR];

    size_t conv_filter_size[] = {K_X, K_Y, 4*GR, GR};
    size_t conv_strides[] = {1, 1};
    int conv_offset[] = {-1, -1};

    for (int i = 0; i < 4*GR; ++i) {
        bn_scale_shift[i] = ((double)(rand()%256)) / 255.f;
        if (bn_scale_shift[i] == 0.f)
            bn_scale_shift[i] = 1.f;

        bn_scale_shift[i + 4*GR] = ((double)(rand()%256 - 128)) / 127.f;
    }

    for (int fout = 0; fout < GR; ++fout)
        for (int fin = 0; fin < 4*GR; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter_param[fout][fin][k_y][k_x] = ((double)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < GR; ++fout)
        conv_bias_param[fout] = ((double)(rand()%256 - 128)) / 127.f;

    // Create the DenseNet block
    dnnLayout_t input_layout;
    CHECK_ERR(dnnLayoutCreate_F64(&input_layout, TENSOR_DIMENSION, input_size, input_strides), err);

    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    dnnPrimitive_t bn_primitive;
    CHECK_ERR(dnnBatchNormalizationCreateForward_F64(&bn_primitive, attributes, input_layout, EPSILON), err);

    dnnPrimitive_t relu_primitive;
    CHECK_ERR(dnnReLUCreateForward_F64(&relu_primitive, attributes, input_layout, 0.f), err);

    dnnPrimitive_t conv_primitive;
    CHECK_ERR(dnnConvolutionCreateForwardBias_F64(&conv_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, input_size, output_size, conv_filter_size, conv_strides, conv_offset, dnnBorderZeros), err);

    // Allocate buffers
    double* input_buf = malloc(sizeof(double) * N * N * 4*GR * BATCH_SIZE);
    double* bn_buf = malloc(sizeof(double) * N * N * 4*GR * BATCH_SIZE);
    double* output_buf = malloc(sizeof(double) * N * N * GR * BATCH_SIZE);

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int z = 0; z < 4*GR; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    input_buf[x + y*N + z*N*N + n*N*N*4*GR] = ((double)(rand() % 256)) / 255.f;

    // Execute the block
    double* resources[dnnResourceNumber] = {0};
    double times[NB_TESTS];
    clock_t start, end;

    for (int i = 0; i < NB_TESTS; ++i) {
        start = clock();

        resources[dnnResourceSrc] = input_buf;
        resources[dnnResourceScaleShift] = bn_scale_shift;
        resources[dnnResourceWorkspace] = bn_workspace;
        resources[dnnResourceDst] = bn_buf;
        CHECK_ERR(dnnExecute_F64(bn_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = bn_buf;
        resources[dnnResourceDst] = bn_buf;
        CHECK_ERR(dnnExecute_F64(relu_primitive, (void**)resources), err);

        resources[dnnResourceSrc] = bn_buf;
        resources[dnnResourceFilter] = (void*)conv_filter_param;
        resources[dnnResourceBias] = conv_bias_param;
        resources[dnnResourceDst] = output_buf;
        CHECK_ERR(dnnExecute_F64(conv_primitive, (void**)resources), err);

        end = clock();
        double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
        times[i] = time_taken;
    }

    printf("\n\n\tDenseNet block time: %f ms.\n", median(NB_TESTS, times));

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
                    fprintf(f, "%.17g\n", output_buf[x + y*N + z*N*N + n*N*N*GR]);

    fclose(f);
    
    free(output_buf);
    free(bn_buf);
    free(input_buf);
    dnnDelete_F64(conv_primitive);
    dnnDelete_F64(relu_primitive);
    dnnDelete_F64(bn_primitive);
    dnnPrimitiveAttributesDestroy_F64(attributes);
    dnnLayoutDelete_F64(input_layout);
    return 0;
}