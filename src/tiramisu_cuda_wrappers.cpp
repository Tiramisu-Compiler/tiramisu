//
// Created by malek on 3/20/18.
//

#include <cuda_runtime.h>
#include <cstdint>
#include <Halide.h>

using size_type = uint64_t;

extern "C"
int tiramisu_cuda_memcpy_to_device(void * to, halide_buffer_t * from, uint64_t size)
{
   cudaMemcpy(to, Halide::Runtime::Buffer<>(*from).data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);
   return 0;
}

extern "C"
int tiramisu_cuda_memcpy_to_host(halide_buffer_t * to, void * from, uint64_t size)
{
    cudaMemcpy(Halide::Runtime::Buffer<>(*to).data(), from, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return 0;
}

extern "C"
void * tiramisu_cuda_malloc(uint64_t size)
{
    void * result;
    cudaMalloc(&result, size);
    return result;
}

extern "C"
int tiramisu_cuda_free(void * ptr)
{
    cudaFree(ptr);
    return 0;
}
