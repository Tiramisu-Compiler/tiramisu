//
// Created by malek on 3/20/18.
//

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include "cublas_v2.h"

using size_type = uint64_t;

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

extern "C"
int tiramisu_cuda_memcpy_to_device(void * to, void * from, uint64_t size)
{
   handle_cuda_error(cudaMemcpy(to, from, size, cudaMemcpyKind::cudaMemcpyHostToDevice), __FUNCTION__);
   return 0;
}

extern "C"
int tiramisu_cuda_memcpy_to_host(void * to, void * from, uint64_t size)
{
    handle_cuda_error(cudaMemcpy(to, from, size, cudaMemcpyKind::cudaMemcpyDeviceToHost), __FUNCTION__);
    return 0;
}

extern "C"
int tiramisu_cuda_memcpy_to_symbol(void * to, void * from, uint64_t size)
{
    handle_cuda_error(cudaMemcpyToSymbol(to, from, size), __FUNCTION__);
    return 0;
}

extern "C"
void * tiramisu_cuda_malloc(uint64_t size)
{
    void * result;
    handle_cuda_error(cudaMalloc(&result, size), __FUNCTION__);
    return result;
}

extern "C"
int tiramisu_cuda_free(void * ptr)
{
    handle_cuda_error(cudaFree(ptr), __FUNCTION__);
    return 0;
}

/**
 * This is a reduced interface for the cuBLAS GEMM API.
 * It multiplies row-major matrices A and B of size MxK and KxN.
 */
extern "C"
int tiramisu_cublas_sgemm(float *A, float *B, float *C,
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
    // The cuBLAS GEMM accepts column major buffers by default. We do a simple
    // trick here to multiply row major matrices. From a row-major perspective,
    // column-major multiplication basically transposes inputs, multiplies, and
    // transposes the output again: cublas(A, B) = ((A^T)x(B^T))^T = BxA
    // So it is actually equivalent to row-major GEMM with inputs swapped.
    // We need to reorder the size parameters as well to make it work:
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

extern "C"
int32_t tiramisu_cuda_stream_synchronize(int32_t dummy)
{
    cudaStreamSynchronize(0);
    return 0;
}

