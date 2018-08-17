GPU Implementation of gemm. From the build directory:
- `make run_benchmark_gpu_gemm` to run the benchmark
- `make run_benchmark_gpu_gemm_nvprof` to see Nvidia profiles of Tiramisu and cublas kernels.
- `make run_benchmark_gpu_gemm_correctness` to run the correctness test. Use smaller sizes to prevent precision errors.

Note that the actual algorithm we are testing is transposed matrix multiplication where matrix B is transposed by default. Inputs and outputs are in row-major format.

Since cublas is column-major by default, we swap inputs to get C in row major format. This assures that the operations are exactly same between Tiramisu and cublas. However, this is not the fastest cublas kernel. If C is made column-major, cublas gets ~10% faster.
