GPU Implementation of gemm
To run: `make`
To run Nvidia Profiler: `make nvprof`

With current implementation kernel time scales with N^3 with respect to input dimensions (so does cublas).
However CPU overhead grows faster thus the tiramisu-cublas margin increases as matrices get bigger.
