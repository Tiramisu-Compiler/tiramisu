GPU Implementation of gemm. From the build directory:
- `make run_benchmark_gpu_gemm` to run the benchmark
- `make run_benchmark_gpu_gemm_nvprof` to see Nvidia profiles of Tiramisu and cublas kernels.

With current implementation kernel time scales with N^3 with respect to input dimensions (so does cublas).
However CPU overhead grows faster thus the tiramisu-cublas margin increases as matrices get bigger.
