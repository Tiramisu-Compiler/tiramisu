#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

    std::vector<double> duration_vector;
    dnnError_t err;

    // Define some parameters
    size_t conv_input_size[] = {N, N, FIn, BATCH_SIZE};
    size_t conv_input_strides[] = {FIn, FIn*N, 1, FIn*N*N};

    size_t output_size[] = {N, N, FOut, BATCH_SIZE};
    size_t output_strides[] = {1, N, N*N, N*N*FOut};

    float conv_filter_param[FOut][FIn][K_Y][K_X];
    float conv_bias_param[FOut];

    size_t conv_filter_size[] = {K_X, K_Y, FIn, FOut};
    size_t conv_filter_strides[] = {1, K_X, K_X*K_Y, K_X*K_Y*FIn};

    size_t conv_strides[] = {1, 1};
    int conv_offset[] = {-1, -1};

    for (int fout = 0; fout < FOut; ++fout)
        for (int fin = 0; fin < FIn; ++fin)
            for (int k_y = 0; k_y < K_Y; ++k_y)
                for (int k_x = 0; k_x < K_X; ++k_x)
                    conv_filter_param[fout][fin][k_y][k_x] = ((float)(rand()%256 - 128)) / 127.f;

    for (int fout = 0; fout < FOut; ++fout)
        conv_bias_param[fout] = ((float)(rand()%256 - 128)) / 127.f;

    // Allocate buffers
    float* input_buf = (float*)malloc(sizeof(float) * FIn * IMG_WIDTH * IMG_HEIGHT * BATCH_SIZE);
    float* resized_buf = (float*)malloc(sizeof(float) * FIn * N * N * BATCH_SIZE);
    float* output_buf;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int y = 0; y < IMG_HEIGHT; ++y)
            for (int x = 0; x < IMG_WIDTH; ++x)
                for (int fin = 0; fin < FIn; ++fin)
                    input_buf[fin + x*FIn + y*FIn*IMG_WIDTH + n*FIn*IMG_WIDTH*IMG_HEIGHT] = ((float)(rand() % 256)) / 255.f;

    // Create Resize-Conv
    float* res_conv[dnnResourceNumber] = {0};

    dnnPrimitiveAttributes_t attributes;
    CHECK_ERR(dnnPrimitiveAttributesCreate_F64(&attributes), err);

    // Create convolution
    dnnLayout_t lt_user_input, lt_user_filt;
    dnnLayout_t lt_conv_input, lt_conv_filt, lt_conv_output;

    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_input, TENSOR_DIMENSION, conv_input_size, conv_input_strides), err);
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt, TENSOR_DIMENSION, conv_filter_size, conv_filter_strides), err);

    dnnPrimitive_t cv_usr_to_conv_input, cv_usr_to_conv_filt;
    dnnPrimitive_t conv_primitive;
    CHECK_ERR(dnnConvolutionCreateForwardBias_F32(&conv_primitive, attributes, dnnAlgorithmConvolutionDirect, TENSOR_DIMENSION, conv_input_size, output_size, conv_filter_size, conv_strides, conv_offset, dnnBorderZeros), err);

    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_input, conv_primitive, dnnResourceSrc), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_filt, conv_primitive, dnnResourceFilter), err);
    CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_output, conv_primitive, dnnResourceDst), err);

    CHECK_ERR(init_conversion(&cv_usr_to_conv_filt, &res_conv[dnnResourceFilter], lt_conv_filt, lt_user_filt, (float*)conv_filter_param), err);
    CHECK_ERR(dnnAllocateBuffer_F32((void **)&res_conv[dnnResourceDst], lt_conv_output), err);

    CHECK_ERR(dnnConversionCreate_F32(&cv_usr_to_conv_input, lt_user_input, lt_conv_input), err);
    CHECK_ERR(dnnAllocateBuffer_F32((void **)&res_conv[dnnResourceSrc], lt_conv_input), err);

    CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv_filt, (void*)conv_filter_param, res_conv[dnnResourceFilter]), err);
    res_conv[dnnResourceBias] = conv_bias_param;

    // Create Conv output conversions
    dnnLayout_t lt_user_output;
    CHECK_ERR(dnnLayoutCreate_F32(&lt_user_output, TENSOR_DIMENSION, output_size, output_strides), err);

    dnnPrimitive_t cv_conv_to_usr_output;
    CHECK_ERR(init_conversion(&cv_conv_to_usr_output, &output_buf, lt_user_output, lt_conv_output, res_conv[dnnResourceDst]), err);

    // Execute the block
    for (int i = 0; i < NB_TESTS; ++i) {
        double start = rtclock();

        // Loop through batch dimension to process each image with OpenCV
        for (int j = 0; j < BATCH_SIZE; ++j) {
            cv::Mat input_mat(IMG_HEIGHT, IMG_WIDTH, CV_32FC3, (uchar*)&input_buf[j * FIn * IMG_WIDTH * IMG_HEIGHT]);
            cv::Mat resized_mat(N, N, CV_32FC3, (uchar*)&resized_buf[j * FIn * N * N]);

            cv::resize(input_mat, resized_mat, {N, N}, 0, 0, cv::INTER_LINEAR);
        }

        CHECK_ERR(dnnConversionExecute_F32(cv_usr_to_conv_input, (void*)resized_buf, res_conv[dnnResourceSrc]), err);
        CHECK_ERR(dnnExecute_F32(conv_primitive, (void**)res_conv), err);

        double end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tResize-Conv block time: "
              << median(duration_vector) << " ms." << std::endl;

    CHECK_ERR(dnnConversionExecute_F32(cv_conv_to_usr_output, res_conv[dnnResourceDst], output_buf), err);

    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 1;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fout = 0; fout < FOut; ++fout)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    fprintf(f, "%.10g\n", output_buf[x + y*N + fout*N*N + n*N*N*FOut]);
        
    fclose(f);
    
    free(input_buf);
    dnnDelete_F64(conv_primitive);

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

    dnnReleaseBuffer_F32(res_conv[dnnResourceSrc]);
    dnnReleaseBuffer_F32(res_conv[dnnResourceFilter]);
    dnnReleaseBuffer_F32(res_conv[dnnResourceDst]);
    dnnReleaseBuffer_F32(output_buf);

    return 0;
}
