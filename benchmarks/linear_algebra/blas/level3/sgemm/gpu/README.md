GPU Implementation of gemm. From the build directory:
- `make run_benchmark_gpu_gemm` to run the benchmark. Compares GPU times.
- `make run_benchmark_gpu_gemm_nvprof` to see Nvidia profiles of Tiramisu and cublas kernels.
- `make run_benchmark_gpu_gemm_correctness` to run the correctness test.

Since cublas is column-major by default, we swap inputs to get C in row major
format. This assures that the operations are exactly same between Tiramisu and
cuBLAS.
