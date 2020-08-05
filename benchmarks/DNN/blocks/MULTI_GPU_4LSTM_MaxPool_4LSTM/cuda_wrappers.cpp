#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <sstream>
#include "cublas_v2.h"

namespace {
    inline void handle_cuda_error(cudaError_t e, const std::string & function_name)
    {
        if (e != cudaError_t::cudaSuccess)
        {
            std::cerr << "Error at " << function_name << ": " << cudaGetErrorString(e) << std::endl;
            exit(1);
        }

    }

    inline void handle_cublas_error(cublasStatus_t e, const std::string &fname)
    {
        if (e != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "Error at " << fname << ". Status: " << e << std::endl;
            exit(1);
        }
    }
}

int32_t wrapper_cuda_set_device(int32_t deviceNumber){
    handle_cuda_error(cudaSetDevice(deviceNumber), __FUNCTION__);
    return 0;
}

int32_t wrapper_cuda_stream_synchronize(int32_t dummy)
{
    cudaStreamSynchronize(0);
    return 0;
}

int wrapper_cublas_sgemm(float *A, float *B, float *C,
                         uint64_t M, uint64_t N, uint64_t K,
                         float alpha, float beta,
                         uint64_t ldA, uint64_t ldB, uint64_t ldC,
                         uint64_t offsetA, uint64_t offsetB, uint64_t offsetC,
                         bool transposeA, bool transposeB)
{
    // TODO: Destroy the handle.
    static bool handle_created = false;
    static cublasHandle_t handle;
    if (!handle_created) {
        cublasCreate(&handle);
        handle_created = true;
    }
    // Default values for tight packing:
    if (ldA == 0) {
        ldA = transposeA ? M : K;
    }
    if (ldB == 0) {
        ldB = transposeB ? K : N;
    }
    if (ldC == 0) {
        ldC = N;
    }
    cublasSetStream(handle, cudaStreamPerThread);
    handle_cublas_error(
        cublasSgemm(handle,
                    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                    N, M, K,
                    &alpha, B + offsetB, ldB, A + offsetA, ldA,
                    &beta, C + offsetC, ldC),
         __FUNCTION__);
    return 0;
}

void * wrapper_cuda_malloc(uint64_t size)
{
    void * result;
    handle_cuda_error(cudaMalloc(&result, size), __FUNCTION__);
    return result;
}

int wrapper_cuda_device_enable_peer_access(int peerDevice, unsigned int flags)
{
  //std::cout << "Peer device : " << peerDevice << std::endl;
  handle_cuda_error(cudaDeviceEnablePeerAccess(peerDevice, flags), __FUNCTION__);
  return 0;
}

int wrapper_cuda_memcpy_peer(float* dst, int dstDevice, float* src, int srcDevice, uint64_t count)
{
  handle_cuda_error(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count * sizeof(float)), __FUNCTION__);
  return 0;
}

int wrapper_cuda_memcpy_to_host(void * to, void * from, uint64_t size)
{
    handle_cuda_error(cudaMemcpyAsync(to, from, size, cudaMemcpyKind::cudaMemcpyDeviceToHost), __FUNCTION__);
    return 0;
}

int wrapper_cuda_memcpy_to_device(void * to, void * from, uint64_t size)
{
    handle_cuda_error(cudaMemcpyAsync(to, from, size, cudaMemcpyKind::cudaMemcpyHostToDevice), __FUNCTION__);
    return 0;
}

int wrapper_cuda_free(void * ptr)
{
    handle_cuda_error(cudaFree(ptr), __FUNCTION__);
    return 0;
}

typedef std::chrono::duration<double,std::milli> duration_t;

float get_time(int32_t dummy)
{
    static auto t0 = std::chrono::high_resolution_clock::now();
    return duration_t(std::chrono::high_resolution_clock::now() - t0).count();
}
