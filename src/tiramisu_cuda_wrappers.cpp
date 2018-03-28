//
// Created by malek on 3/20/18.
//

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

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
