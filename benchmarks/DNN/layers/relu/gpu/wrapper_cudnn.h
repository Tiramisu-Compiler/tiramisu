#include <cudnn.h>
#include <cuda.h>
#include <stdio.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
   }
}

// Global variables


void *x;
void *y;


cudnnHandle_t cudnnHandle;
cudnnActivationDescriptor_t activationDesc;
cudnnTensorDescriptor_t *xDesc, *yDesc;

void *workspace;
size_t workSize;

void setup_cudnn(int N, int FIn, int BATCH_SIZE, int NEGATIVE_SLOPES,
                    float *raw_input, float *raw_output) {

    // -------------------------
    // Create cudnn context
    // -------------------------
    cudnnErrCheck(cudnnCreate(&cudnnHandle));

    // -------------------------
    // Set up inputs and outputs
    // -------------------------

    cudaErrCheck(cudaMalloc((void**)&x, N * N * FIn * BATCH_SIZE * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&y, ((N - K_X + 2 * P_X) / S_X + 1)* ((N - K_Y + 2 * P_Y) / S_Y + 1)* FIn * BATCH_SIZE * sizeof(float)));

    //-----------------------
    //init data 
    //-----------------------

    CUDA_CALL(cudaMemcpy(x, raw_input, N * N * FIn * BATCH_SIZE * sizeof(float),cudaMemcpyHostToDevice));

    //------------------------
    // Tensor desc
    //-------------------------

    xDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    yDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));

    cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc));

    int dimA[4];
    int strideA[4];

    dimA[0] = BATCH_SIZE;
    dimA[1] = FIn;
    dimA[2] = N;
    dimA[3] = N;

    strideA[0] = 1;
    strideA[1] = 1;
    strideA[2] = S_X;
    strideA[3] = S_Y;

    cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 4, dimA, strideA));

    dimA[0] = BATCH_SIZE;
    dimA[1] = FIn;
    dimA[2] = N;
    dimA[3] = N;
    
    strideA[0] = 1;
    strideA[1] = 1;
    strideA[2] = S_X;
    strideA[3] = S_Y;

    cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 4, dimA, strideA));


    // -------------------------
    // Set up the Pooling descriptor
    // -------------------------


    cudnnActivationMode_t activationMode;
    cudnnActivationAlgo activationAlgo;



    activationMode = CUDNN_ACTIVATION_RELU;


    cudnnErrCheck(cudnnCreateActivationDescriptor(&activationDesc));

    cudnnErrCheck(cudnnSetActivationDescriptor(
                activationDesc,
                CUDNN_ACTIVATION_RELU,
                CUDNN_PROPAGATE_NAN,
                1));


     cudnnActivationMode_t *mode;
     cudnnNanPropagation_t *reluNanOpt;
     double f;

     cudnnErrCheck(cudnnGetActivationDescriptor(
                activationDesc,
                mode,
                reluNanOpt,
                f));
}
float run_cudnn() {
    cudaErrCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float timeForward;
    float alpha = 1.f;
    float beta = 0.f;

    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));


    cudnnErrCheck(cudnnActivationForward(
                cudnnHandle,
                acivationDesc,
                alpha,
                xDesc,
                x,
                beta,
                yDesc,
                y));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));
    cudaErrCheck(cudaMemcpy(raw_output, y, (N - K_X + 2 * P_X) / (S_X + 1)* (N - K_Y + 2 * P_Y) / (S_Y + 1)* FIn * BATCH_SIZE * sizeof(float),cudaMemcpyDeviceToHost));

    // Make double-sure everything is finished before we copy for result checking.
    cudaDeviceSynchronize();

    return timeForward;
}
void free_cudnn() {
    cudaFree(x);
    cudaFree(y);
    cudaFree(workspace);
    cudnnDestroy(cudnnHandle);
}

