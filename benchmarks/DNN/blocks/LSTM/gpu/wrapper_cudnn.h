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
int seqLength;

void *x;
void *hx = NULL;
void *cx = NULL;

void *y;
void *hy = NULL;
void *cy = NULL;

void *w;

cudnnHandle_t cudnnHandle;
cudnnRNNDescriptor_t rnnDesc;
cudnnTensorDescriptor_t *xDesc, *yDesc;
cudnnTensorDescriptor_t hxDesc, cxDesc;
cudnnTensorDescriptor_t hyDesc, cyDesc;
cudnnFilterDescriptor_t wDesc;

void *workspace;
size_t workSize;

void setup_cudnn(int _seqLength, int numLayers, int batch_size, int feature_size,
        float *raw_Weights, float *raw_biases, float *raw_x) {
    seqLength = _seqLength;
    int hiddenSize = feature_size;
    int inputSize = hiddenSize;
    int miniBatch = batch_size;

    // -------------------------
    // Create cudnn context
    // -------------------------
    cudnnErrCheck(cudnnCreate(&cudnnHandle));

    // -------------------------
    // Set up inputs and outputs
    // -------------------------
    cudaErrCheck(cudaMalloc((void**)&x, seqLength * inputSize * miniBatch * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&y, seqLength * hiddenSize * miniBatch * sizeof(float)));

    xDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    yDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));

    int dimA[3];
    int strideA[3];

    for (int i = 0; i < seqLength; i++) {
        cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));

        dimA[0] = miniBatch;
        dimA[1] = inputSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = miniBatch;
        dimA[1] = hiddenSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
    }


    dimA[0] = numLayers;
    dimA[1] = miniBatch;
    dimA[2] = hiddenSize;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // -------------------------
    // Set up the dropout descriptor (needed for the RNN descriptor)
    // -------------------------

    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));

    size_t stateSize;
    void *states;
    cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

    cudaErrCheck(cudaMalloc(&states, stateSize));

    cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc,
                cudnnHandle,
                0,
                states,
                stateSize,
                0));

    // -------------------------
    // Set up the RNN descriptor
    // -------------------------
    cudnnRNNMode_t RNNMode;
    cudnnRNNAlgo_t RNNAlgo;

    cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));

    RNNMode = CUDNN_LSTM;

    RNNAlgo = CUDNN_RNN_ALGO_STANDARD;

    cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle,
                rnnDesc,
                hiddenSize,
                numLayers,
                dropoutDesc,
                CUDNN_LINEAR_INPUT,
                CUDNN_UNIDIRECTIONAL,
                RNNMode,
                RNNAlgo,
                CUDNN_DATA_FLOAT));


    // -------------------------
    // Set up parameters
    // -------------------------
    // This needs to be done after the rnn descriptor is set as otherwise
    // we don't know how many parameters we have to allocate

    cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));

    size_t weightsSize;
    cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));

    int dimW[3];
    dimW[0] =  weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;

    cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

    cudaErrCheck(cudaMalloc((void**)&w,  weightsSize));


    // -------------------------
    // Set up work space and reserved memory
    // -------------------------

    // Need for every pass
    cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, &workSize));
    cudaErrCheck(cudaMalloc((void**)&workspace, workSize));

    // *********************************************************************************************************
    // Initialise inputs
    // *********************************************************************************************************
    // initGPUData((float*)x, seqLength * inputSize * miniBatch, 1.f);

    // Weights
    int numLinearLayers = 8;

    for (int layer = 0; layer < numLayers; layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            float *linLayerMat;

            cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle,
                        rnnDesc,
                        layer,
                        xDesc[0],
                        wDesc,
                        w,
                        linLayerID,
                        linLayerMatDesc,
                        (void**)&linLayerMat));

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];
            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                        3,
                        &dataType,
                        &format,
                        &nbDims,
                        filterDimA));

            // initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

            cudnnFilterDescriptor_t linLayerBiasDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            float *linLayerBias;

            cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle,
                        rnnDesc,
                        layer,
                        xDesc[0],
                        wDesc,
                        w,
                        linLayerID,
                        linLayerBiasDesc,
                        (void**)&linLayerBias));

            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                        3,
                        &dataType,
                        &format,
                        &nbDims,
                        filterDimA));

            // initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
        }
    }
}

float run_cudnn() {
    cudaErrCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float timeForward;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
                rnnDesc,
                seqLength,
                xDesc,
                x,
                hxDesc,
                hx,
                cxDesc,
                cx,
                wDesc,
                w,
                yDesc,
                y,
                hyDesc,
                hy,
                cyDesc,
                cy,
                workspace,
                workSize));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));

    // Make double-sure everything is finished before we copy for result checking.
    cudaDeviceSynchronize();

    return timeForward;
}

void free_cudnn() {
    cudaFree(x);
    cudaFree(y);
    cudaFree(workspace);
    cudaFree(w);

    cudnnDestroy(cudnnHandle);
}

