GPU Implementation of gemm. From the build directory:
- `make run_benchmark_gpu_gemm` to run the benchmark. Compares GPU times.
- `make run_benchmark_gpu_gemm_nvprof` to see Nvidia profiles of Tiramisu and cublas kernels.
- `make run_benchmark_gpu_gemm_correctness` to run the correctness test.

Since cublas is column-major by default, we swap inputs to get C in row major
format. This assures that the operations are exactly same between Tiramisu and
cuBLAS.

There are two different schedules implemented in files `generator1.cpp` and
`generator2.cpp`. `generator1` uses alternates between two shared memory buffer
spaces at each iteration, while `generator2` uses two step global->shared copy
with single shared buffer as explained in the Diesel paper. There is also a
simple implementation that uses `cache_shared` API in `generator3`. You can
switch between implementations via `CMakeLists.txt` file.
