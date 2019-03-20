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
                          int M, int N, int K,
                          float alpha, float beta)
{
    // TODO: Destroy the handle.
    static bool handle_created = false;
    static cublasHandle_t handle;
    if (!handle_created) {
        cublasCreate(&handle);
        handle_created = true;
    }
    // The cuBLAS GEMM accepts column major buffers by default. We do a simple
    // trick here to multiply row major matrices. From a row-major perspective,
    // column-major multiplication basically transposes inputs, multiplies, and
    // transposes the output again: cublas(A, B) = ((A^T)x(B^T))^T = BxA
    // So it is actually equivalent to row-major GEMM with inputs swapped.
    // We need to reorder the size parameters as well to make it work:
    handle_cublas_error(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, B, N, A, K, &beta, C, N),
         __FUNCTION__);
    return 0;
}
