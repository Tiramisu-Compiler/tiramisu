#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <algorithm>

#include "multi_gpu_deepspeech_kernels.cu"
#include "cuda_wrappers.cpp"
#include "configuration.h"
#include "cudnn_functions.cu"

#define GPU1_ID 0
#define GPU2_ID 1
#define GPU3_ID 2
#define ENABLE_PEER_ACCESS 1

// Buffer sizes
#define WEIGHTS_SIZE (2 * NUM_LAYERS * 4 * FEATURE_SIZE * FEATURE_SIZE * 4)
#define BIASES_SIZE (4 * FEATURE_SIZE * NUM_LAYERS * 4)

#define WEIGHTS2_SIZE (2 * NUM_LAYERS * 4 * FEATURE_SIZE * FEATURE_SIZE * 4)
#define BIASES2_SIZE (4 * FEATURE_SIZE * NUM_LAYERS * 4)

#define X_SIZE (FEATURE_SIZE * BATCH_SIZE * SEQ_LENGTH * 4)
#define Y_SIZE (FEATURE_SIZE * BATCH_SIZE * (SEQ_LENGTH / 2) * 4)

#define TMP_SIZE (SEQ_LENGTH * BATCH_SIZE * 4 * FEATURE_SIZE * 4)
#define TMP2_SIZE ((SEQ_LENGTH / 2) * BATCH_SIZE * 4 * FEATURE_SIZE * 4)

#define H_SIZE ((NUM_LAYERS + 1)* (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE * 4)
#define H2_SIZE ((NUM_LAYERS + 1)* (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE * 4)

#define C_SIZE ((NUM_LAYERS)* (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE * 4)
#define C2_SIZE ((NUM_LAYERS)* (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE * 4)

#define NB_TIMES 100
bool first_execution = true;
using namespace std;

double median(std::vector<double> scores)
{
    double median;
    size_t size = scores.size();

    sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
    }
    else
    {
        median = scores[size / 2];
    }

    return median;
}
std::vector<double> duration_vector, duration_vector2;

void deepspeech2(float *buf_Weights_cpu, float *buf_biases_cpu,
                 float *buf_Weights2_cpu, float *buf_biases2_cpu,
                 float *buf_x_cpu, float *buf_y_cpu,
                 float *time_start, float *time_end)
                 {

/** GPU1 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU1_ID);
    if (first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU2_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU3_ID, 0);
    }
    float *buf_x_gpu1, *buf_weights_gpu1, *buf_biases_gpu1, *buf_tmp_gpu1, *buf_weights_T_gpu1, *buf_h_gpu1, *buf_c_gpu1;
    buf_x_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)X_SIZE);
    buf_weights_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS_SIZE);
    buf_biases_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)BIASES_SIZE);
    buf_tmp_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)TMP_SIZE);
    buf_weights_T_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS_SIZE);
    buf_h_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)H_SIZE);
    buf_c_gpu1 = (float *) wrapper_cuda_malloc((uint64_t)C_SIZE);

    wrapper_cuda_memcpy_to_device(buf_weights_gpu1, buf_Weights_cpu, (uint64_t)WEIGHTS_SIZE);
    wrapper_cuda_memcpy_to_device(buf_biases_gpu1, buf_biases_cpu, (uint64_t)BIASES_SIZE);
    cudaMemset(buf_tmp_gpu1, 0, TMP_SIZE);

/** GPU2 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU2_ID);
    if(first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU1_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU3_ID, 0);
    }
    float *buf_weights_gpu2, *buf_biases_gpu2, *buf_tmp_gpu2, *buf_weights_T_gpu2, *buf_h_gpu2, *buf_c_gpu2, *buf_weights2_gpu2, *buf_biases2_gpu2, *buf_weights2_T_gpu2;
    float *buf_h2_gpu2, *buf_c2_gpu2, *buf_tmp2_gpu2;
    buf_weights_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS_SIZE);
    buf_biases_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)BIASES_SIZE);
    buf_tmp_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)TMP_SIZE);

    buf_weights_T_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS_SIZE);
    buf_h_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)H_SIZE);
    buf_c_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)C_SIZE);

    buf_weights2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS2_SIZE);
    buf_biases2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)BIASES2_SIZE);
    buf_weights2_T_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS2_SIZE);

    buf_tmp2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)TMP2_SIZE);
    buf_h2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)H2_SIZE);
    buf_c2_gpu2 = (float *) wrapper_cuda_malloc((uint64_t)C2_SIZE);

    wrapper_cuda_memcpy_to_device(buf_weights_gpu2, buf_Weights_cpu, (uint64_t)WEIGHTS_SIZE);
    wrapper_cuda_memcpy_to_device(buf_biases_gpu2, buf_biases_cpu, (uint64_t)BIASES_SIZE);
    wrapper_cuda_memcpy_to_device(buf_weights2_gpu2, buf_Weights2_cpu, (uint64_t)WEIGHTS2_SIZE);
    wrapper_cuda_memcpy_to_device(buf_biases2_gpu2, buf_biases2_cpu, (uint64_t)BIASES2_SIZE);
    cudaMemset(buf_tmp_gpu2, 0, TMP_SIZE);
    cudaMemset(buf_tmp2_gpu2, 0, TMP2_SIZE);
/** GPU3 INITIALIZATIONS */
    wrapper_cuda_set_device(GPU3_ID);
    float *buf_y_gpu = (float *) wrapper_cuda_malloc((uint64_t)Y_SIZE);

    if(first_execution && ENABLE_PEER_ACCESS){
      wrapper_cuda_device_enable_peer_access(GPU1_ID, 0);
      wrapper_cuda_device_enable_peer_access(GPU2_ID, 0);
    }

    float *buf_weights2_gpu3, *buf_biases2_gpu3, *buf_tmp2_gpu3, *buf_weights2_T_gpu3, *buf_h2_gpu3, *buf_c2_gpu3;
    buf_weights2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS2_SIZE);
    buf_biases2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)BIASES2_SIZE);
    buf_tmp2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)TMP2_SIZE);

    buf_weights2_T_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)WEIGHTS2_SIZE);
    buf_h2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)H2_SIZE);
    buf_c2_gpu3 = (float *) wrapper_cuda_malloc((uint64_t)C2_SIZE);

    wrapper_cuda_memcpy_to_device(buf_weights2_gpu3, buf_Weights2_cpu, (uint64_t)WEIGHTS2_SIZE);
    wrapper_cuda_memcpy_to_device(buf_biases2_gpu3, buf_biases2_cpu, (uint64_t)BIASES2_SIZE);
    cudaMemset(buf_tmp2_gpu3, 0, TMP2_SIZE);

    time_start[0] = get_time(0);

    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_memcpy_to_device(buf_x_gpu1, buf_x_cpu, (uint64_t)X_SIZE);

    _kernel_0_wrapper(buf_weights_gpu1, buf_weights_T_gpu1);
    _kernel_1_wrapper(buf_h_gpu1);
    _kernel_2_wrapper(buf_c_gpu1);
    _kernel_3_wrapper(buf_h_gpu1, buf_x_gpu1);

    wrapper_cuda_set_device(GPU2_ID);
    _kernel_4_wrapper(buf_weights_gpu2, buf_weights_T_gpu2);
    _kernel_5_wrapper(buf_weights2_T_gpu2, buf_weights2_gpu2);
    _kernel_6_wrapper(buf_h_gpu2);
    _kernel_7_wrapper(buf_c_gpu2);
    _kernel_8_wrapper(buf_h2_gpu2);
    _kernel_9_wrapper(buf_c2_gpu2);

    wrapper_cuda_set_device(GPU3_ID);
    _kernel_10_wrapper(buf_weights2_gpu3, buf_weights2_T_gpu3);
    _kernel_11_wrapper(buf_h2_gpu3);
    _kernel_12_wrapper(buf_c2_gpu3);

    for(int c1=0; c1<(SEQ_LENGTH/GEMM_BATCH) / 2 + 2 * NUM_LAYERS + 1; c1++){
         #pragma omp parallel for
         for(int c3  = max(c1-(SEQ_LENGTH/GEMM_BATCH) / 2 + 1, 0); c3 < max(c1-(SEQ_LENGTH/GEMM_BATCH) / 2 + 1, 0) + min(c1, 2 * NUM_LAYERS) - max(c1-(SEQ_LENGTH/GEMM_BATCH) / 2 + 1, 0) +1; c3++){
            if(c3 < 3){
                wrapper_cuda_set_device(GPU1_ID);

                for (int c5 = 0; c5 < 2; c5++) {
                    wrapper_cublas_sgemm(buf_h_gpu1, buf_weights_T_gpu1, buf_tmp_gpu1,
                      (uint64_t)(GEMM_BATCH * BATCH_SIZE), (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                      1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                      uint64_t(((((c3*(SEQ_LENGTH + 1)) + ((((c1 - c3)*2) + c5)*GEMM_BATCH))*BATCH_SIZE * FEATURE_SIZE) + BATCH_SIZE * FEATURE_SIZE)),
                      uint64_t((c3 * 2 * 4 * FEATURE_SIZE * FEATURE_SIZE)),
                      uint64_t(((((c1 - c3)*2) + c5) * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE)),
                      (uint32_t)0, (uint32_t)0
                    );

                    for (int c7 = 0; c7 < GEMM_BATCH; c7++) {
                        wrapper_cublas_sgemm(buf_h_gpu1, buf_weights_T_gpu1, buf_tmp_gpu1,
                          (uint64_t)BATCH_SIZE, (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                          1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                          uint64_t(((((c3*(SEQ_LENGTH + 1)) + (((((c1 - c3)*2) + c5)*GEMM_BATCH) + c7))* BATCH_SIZE * FEATURE_SIZE) + (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE)),
                          uint64_t(((c3*2 * 4 * FEATURE_SIZE * FEATURE_SIZE) + 4 * FEATURE_SIZE * FEATURE_SIZE)),
                          uint64_t(((((((c1 - c3)*2) + c5)*GEMM_BATCH) + c7)* BATCH_SIZE * 4 * FEATURE_SIZE)),
                          (uint32_t)0, (uint32_t)0
                        );
                        _kernel_13_wrapper(c1, c3, c5, c7, buf_biases_gpu1, buf_c_gpu1, buf_h_gpu1, buf_tmp_gpu1);
                    }
                    wrapper_cuda_stream_synchronize(0);
                }
            }

            // Memcpy GPU0 to GPU1
            #define offset1 ((c3 + 1) * (SEQ_LENGTH + 1) * BATCH_SIZE * FEATURE_SIZE + (2 * (c1 - c3) * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define count1 (2 * GEMM_BATCH * BATCH_SIZE * FEATURE_SIZE)
            if (c3 == 2 && ENABLE_PEER_ACCESS)
              wrapper_cuda_memcpy_peer((buf_h_gpu2 + offset1), GPU2_ID, (buf_h_gpu1 + offset1), GPU1_ID, (uint64_t) count1);

            if (c3 == 3)
              wrapper_cuda_stream_synchronize(0);

            if (((c3 < 6) && (2 < c3)))
                wrapper_cuda_set_device(GPU2_ID);

            if ((c3 == 3))
              for (int c5 = 0; c5 < 2; c5++) {
                  wrapper_cublas_sgemm(buf_h_gpu2, buf_weights_T_gpu2, buf_tmp_gpu2,
                    (uint64_t)(GEMM_BATCH * BATCH_SIZE), (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                    1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                    uint64_t(((((c1*2) + c5)*GEMM_BATCH*BATCH_SIZE*FEATURE_SIZE) + (3 * (SEQ_LENGTH + 1) - 3 * 2 * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)),
                    (uint64_t)(3 * 2 * 4 * FEATURE_SIZE * FEATURE_SIZE),
                    uint64_t(((((c1*2) + c5) * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE) - (3 * 2 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE))),
                    (uint32_t)0, (uint32_t)0
                  );

                  for (int c7 = 0; c7 < GEMM_BATCH; c7++) {
                      wrapper_cublas_sgemm(buf_h_gpu2, buf_weights_T_gpu2, buf_tmp_gpu2,
                        (uint64_t)BATCH_SIZE, (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                        1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                        uint64_t(((((((c1*2) + c5)*GEMM_BATCH) + c7)*BATCH_SIZE * FEATURE_SIZE) + (4 * (SEQ_LENGTH + 1) - 3 * 2 * GEMM_BATCH) * BATCH_SIZE * FEATURE_SIZE)),
                        (uint64_t)(7 * 4 * FEATURE_SIZE * FEATURE_SIZE),
                        uint64_t(((((((c1*2) + c5)*GEMM_BATCH) + c7)*BATCH_SIZE * 4 * FEATURE_SIZE) - 3 * 2 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE)),
                        (uint32_t)0, (uint32_t)0
                      );
                      _kernel_14_wrapper(c1, c3, c5, c7, buf_biases_gpu2, buf_c_gpu2, buf_h_gpu2, buf_tmp_gpu2);
                  }
                  wrapper_cuda_stream_synchronize(0);
              }

            if ((c3 == 4))
              _kernel_15_wrapper(c1, c3, buf_h_gpu2, buf_h2_gpu2);

            if (c3 == 5)
              wrapper_cuda_stream_synchronize(0);

            if ((c3 == 5)) {
              wrapper_cublas_sgemm(buf_h2_gpu2, buf_weights2_T_gpu2, buf_tmp2_gpu2,
                (uint64_t)(GEMM_BATCH * BATCH_SIZE), (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                uint64_t(((c1*GEMM_BATCH*BATCH_SIZE*FEATURE_SIZE) - (5 * GEMM_BATCH - 1) * BATCH_SIZE * FEATURE_SIZE)),
                (uint64_t)0,
                uint64_t(((c1 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE) - 5 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE)),
                (uint32_t)0, (uint32_t)0
              );

              for (int c5 = 0; c5 < GEMM_BATCH; c5++) {
                  wrapper_cublas_sgemm(buf_h2_gpu2, buf_weights2_T_gpu2, buf_tmp2_gpu2,
                    (uint64_t)BATCH_SIZE, (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                    1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                    uint64_t(((((c1 * GEMM_BATCH) + c5)*BATCH_SIZE*FEATURE_SIZE) + (SEQ_LENGTH/2 + 1 - 5 * GEMM_BATCH) * BATCH_SIZE * FEATURE_SIZE)),
                    (uint64_t)(4 * FEATURE_SIZE * FEATURE_SIZE),
                    uint64_t(((((c1 * GEMM_BATCH) + c5) * BATCH_SIZE * 4 * FEATURE_SIZE) - 5 * GEMM_BATCH * BATCH_SIZE * 4 * FEATURE_SIZE)),
                    (uint32_t)0, (uint32_t)0
                  );

                  _kernel_16_wrapper(c1, c3, c5, buf_biases2_gpu2, buf_c2_gpu2, buf_h2_gpu2, buf_tmp2_gpu2);
              }
            }

            // Memcpy GPU0 to GPU1
            #define offset2 ((c3 + 1 - 5) * (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE + ((c1 - c3) * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define offset3 ((c3 + 1 - 5) * (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE + ((c1 - c3) * GEMM_BATCH + 1) * BATCH_SIZE * FEATURE_SIZE)
            #define count2_3 (GEMM_BATCH * BATCH_SIZE * FEATURE_SIZE)
            if (c3 == 5 && ENABLE_PEER_ACCESS)
                wrapper_cuda_memcpy_peer((buf_h2_gpu3 + offset3), GPU3_ID, (buf_h2_gpu2 + offset2), GPU2_ID, (uint64_t) count2_3);

            if (c3 == 6)
              wrapper_cuda_stream_synchronize(0);

            if ((5 < c3)) {
                wrapper_cuda_set_device(GPU3_ID);
                wrapper_cublas_sgemm(buf_h2_gpu3, buf_weights2_T_gpu3, buf_tmp2_gpu3,
                  (uint64_t)(BATCH_SIZE * GEMM_BATCH), (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                  1.000000f, 0.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                  uint64_t((((c3*(SEQ_LENGTH/2 + 1)) + ((c1 - c3)*GEMM_BATCH))*BATCH_SIZE * FEATURE_SIZE - (5 * (SEQ_LENGTH/2 + 1) - 1) * BATCH_SIZE * FEATURE_SIZE)),
                  uint64_t(((c3 * 2 * 4 * FEATURE_SIZE * FEATURE_SIZE) - (5 * 2 * 4 * FEATURE_SIZE * FEATURE_SIZE))),
                  uint64_t(((c1 - c3)*GEMM_BATCH*BATCH_SIZE*4*FEATURE_SIZE)),
                  (uint32_t)0, (uint32_t)0
                );

                for (int c5 = 0; c5 < GEMM_BATCH; c5++) {
                    wrapper_cublas_sgemm(buf_h2_gpu3, buf_weights2_T_gpu3, buf_tmp2_gpu3,
                      (uint64_t)BATCH_SIZE, (uint64_t)(4 * FEATURE_SIZE), (uint64_t)FEATURE_SIZE,
                      1.000000f, 1.000000f, (uint64_t)0, (uint64_t)0, (uint64_t)0,
                      uint64_t(((((c3*(SEQ_LENGTH/2 + 1)) + (((c1 - c3)*GEMM_BATCH) + c5))*BATCH_SIZE*FEATURE_SIZE) - (4 * (SEQ_LENGTH/2 + 1) * BATCH_SIZE * FEATURE_SIZE))),
                      uint64_t(((c3 * 2 * 4 * FEATURE_SIZE * FEATURE_SIZE) - (9 * 4 * FEATURE_SIZE * FEATURE_SIZE))),
                      uint64_t(((((c1 - c3) * GEMM_BATCH) + c5) * BATCH_SIZE * 4 * FEATURE_SIZE)),
                      (uint32_t)0, (uint32_t)0
                    );
                    _kernel_17_wrapper(c1, c3, c5, buf_biases2_gpu3, buf_c2_gpu3, buf_h2_gpu3, buf_tmp2_gpu3);
                }
            }
        }
    }
    wrapper_cuda_set_device(GPU3_ID);
    _kernel_18_wrapper(buf_y_gpu, buf_h2_gpu3);
    wrapper_cuda_stream_synchronize(0);
    wrapper_cuda_memcpy_to_host(buf_y_cpu, buf_y_gpu, (uint64_t)Y_SIZE);

    //// Wait for all threads
    wrapper_cuda_set_device(GPU3_ID);
    wrapper_cuda_stream_synchronize(0);

    wrapper_cuda_set_device(GPU2_ID);
    wrapper_cuda_stream_synchronize(0);

    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_stream_synchronize(0);

/////////////// Free buffers
    wrapper_cuda_set_device(GPU1_ID);
    wrapper_cuda_free(buf_x_gpu1);
    wrapper_cuda_free(buf_weights_gpu1);
    wrapper_cuda_free(buf_biases_gpu1);
    wrapper_cuda_free(buf_tmp_gpu1);
    wrapper_cuda_free(buf_weights_T_gpu1);
    wrapper_cuda_free(buf_h_gpu1);
    wrapper_cuda_free(buf_c_gpu1);

    wrapper_cuda_set_device(GPU2_ID);
    wrapper_cuda_free(buf_weights_gpu2);
    wrapper_cuda_free(buf_biases_gpu2);
    wrapper_cuda_free(buf_tmp_gpu2);
    wrapper_cuda_free(buf_weights_T_gpu2);
    wrapper_cuda_free(buf_h_gpu2);
    wrapper_cuda_free(buf_c_gpu2);
    wrapper_cuda_free(buf_weights2_gpu2);
    wrapper_cuda_free(buf_biases2_gpu2);
    wrapper_cuda_free(buf_weights2_T_gpu2);
    wrapper_cuda_free(buf_h2_gpu2);
    wrapper_cuda_free(buf_c2_gpu2);
    wrapper_cuda_free(buf_tmp2_gpu2);

    wrapper_cuda_set_device(GPU3_ID);
    wrapper_cuda_free(buf_weights2_gpu3);
    wrapper_cuda_free(buf_biases2_gpu3);
    wrapper_cuda_free(buf_tmp2_gpu3);
    wrapper_cuda_free(buf_weights2_T_gpu3);
    wrapper_cuda_free(buf_h2_gpu3);
    wrapper_cuda_free(buf_c2_gpu3);
    wrapper_cuda_free(buf_y_gpu);

    time_end[0] = get_time(0);
    first_execution=false;
}

int main(int argc, char *argv[]){

    float *buf_Weights_cpu, *buf_biases_cpu,
          *buf_Weights2_cpu, *buf_biases2_cpu,
          *buf_x_cpu, *buf_y_cpu,
          *time_start, *time_end;
    buf_Weights_cpu = (float *)malloc(WEIGHTS_SIZE);
    buf_biases_cpu = (float *)malloc(BIASES_SIZE);

    buf_Weights2_cpu = (float *)malloc(WEIGHTS2_SIZE);
    buf_biases2_cpu = (float *)malloc(BIASES2_SIZE);

    buf_x_cpu = (float *)malloc(X_SIZE);
    buf_y_cpu = (float *)malloc(Y_SIZE);

    time_start = (float *)malloc(sizeof(float));
    time_end = (float *)malloc(sizeof(float));

    /// Data used for debugging
    float *before_maxpool, *after_maxpool;
    before_maxpool = (float *) malloc(X_SIZE);
    after_maxpool = (float *) malloc(Y_SIZE);
    // Initializing the weights and the input
    std::srand(0);
    for (int i = 0; i < WEIGHTS_SIZE/4; i++)
      buf_Weights_cpu[i] = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < BIASES_SIZE/4; i++)
      buf_biases_cpu[i] = (std::rand() % 200 - 100) / 100.;

    for (int i = 0; i < WEIGHTS2_SIZE/4; i++)
      buf_Weights2_cpu[i] = (std::rand() % 200 - 100) / 100.;
    for (int i = 0; i < BIASES2_SIZE/4; i++)
      buf_biases2_cpu[i] = (std::rand() % 200 - 100) / 100.;

    for (int i = 0; i < X_SIZE/4; i++)
      buf_x_cpu[i] = (std::rand() % 200 - 100) / 100.;

    std::cout << "==================================================" << std::endl;
    std::cout << "===============   MULTI GPU  =====================" << std::endl;
    std::cout << "==================================================" << std::endl;

    for (int i = 0; i<NB_TIMES; i++){
      deepspeech2(buf_Weights_cpu, buf_biases_cpu,
                  buf_Weights2_cpu, buf_biases2_cpu,
                  buf_x_cpu, buf_y_cpu,
                  time_start, time_end);

      duration_vector.push_back(time_end[0] - time_start[0]);
    }

    std::cout << "Results " << std::endl;
    std::cout << "-----------------------------" << std::endl;

    std::cout << "Time Median : " << median(duration_vector) << " ms" << std::endl;
    for (int j = 0; j < 10; j++)
      std::cout << buf_y_cpu[j] << ", ";
    std::cout << std::endl << std::endl << std::endl;

    std::cout << "=====================================================" << std::endl;
    std::cout << "================  CUDNN SINGLE GPU  =================" << std::endl;
    std::cout << "=====================================================" << std::endl;

    float *buf_y_cpu_cudnn = (float *)malloc(Y_SIZE);
    wrapper_cuda_set_device(0);

    // Creating the first 4LSTM layers operator
    setup_cudnn(SEQ_LENGTH, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE);

    // Creating the last 4LSTM layers operator
    setup_cudnn2(SEQ_LENGTH/2, NUM_LAYERS, BATCH_SIZE, FEATURE_SIZE);

    float time_lstm1, time_lstm2;
    for (int i = 0; i<NB_TIMES; i++){
      // Running first 4LSTM layers
      time_lstm1 = run_cudnn(buf_Weights_cpu, buf_biases_cpu, buf_x_cpu, before_maxpool);
      time_start[0] = get_time(0);
      // Running downsampling (maxpool)
      downsampling_maxpool_wrapper();
      time_end[0] = get_time(0);
      // Running last 4LSTM layers
      time_lstm2 = run_cudnn2(buf_Weights2_cpu, buf_biases2_cpu, after_maxpool, buf_y_cpu_cudnn);

      duration_vector2.push_back(time_end[0]-time_start[0] + time_lstm1 + time_lstm2);
    }

    std::cout << "Results " << std::endl;
    std::cout << "-----------------------------" << std::endl;

    std::cout << "Time Median : " << median(duration_vector2) << " ms"<<std::endl;
    for (int j = 0; j < 10; j++)
      std::cout << buf_y_cpu_cudnn[j] << ", ";
    std::cout << std::endl;
    if (CHECK_ERROR_DIFFERENCE){
      std::cout << "Testing cudnn-Tiramisu output difference" << std::endl;
      for (int i = 0; i < SEQ_LENGTH/2; i++) {
        DATA_TYPE max_error = 0;
        DATA_TYPE mean_error = 0;
        for (int j = 0; j < BATCH_SIZE; j++) {
          for (int k = 0; k < FEATURE_SIZE; k++) {
            DATA_TYPE res = buf_y_cpu[(i * BATCH_SIZE + j) * FEATURE_SIZE + k];
            DATA_TYPE cuda_res = buf_y_cpu_cudnn[(i * BATCH_SIZE + j) * FEATURE_SIZE + k];
            DATA_TYPE error = std::abs(res - cuda_res);
            if (error > max_error) {
              max_error = error;
            }
            mean_error += error;
          }
        }
        mean_error/= (BATCH_SIZE * FEATURE_SIZE);
        std::cout << "Sequence #" << i << std::endl << "\t\tMean of absolute errors : " << mean_error << std::endl<< "\t\tMax difference: " << max_error << std::endl << std::endl;
      }
    }

    free_cudnn();
    free_cudnn2();

    return 0;
}
