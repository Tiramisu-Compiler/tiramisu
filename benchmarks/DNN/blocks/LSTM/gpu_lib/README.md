LSTM implementation in Tiramisu. The network is the same as in Nvidia blogpost:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/

It uses cuBLAS as backend for GEMM's in the network, and achieves parallelism
internally through CUDA streams API.

Some results on a Kepler 80 GPU vs cuDNN 7:

|                 | Tiramisu | cuDNN7  |
|-----------------|----------|---------|
| 4 layers float  | 57.9ms   | 57.1ms  |
| 8 layers float  | 112.1ms  | 127.0ms |
| 4 layers double | 106.7ms  | 111.8ms |
| 8 layers double | 215.4ms  | 234.9ms |

To run the benchmark you need to have cuDNN installed on the machine.
Set `USE_GPU` and `USE_CUDNN` to true in `configure.cmake`, and set
`CUDNN_LOCATION` to cuDNN location.

From Tiramisu build directory:
- `make run_benchmark_lstm_gpu_lib` to run the benchmark
- `make run_benchmark_lstm_gpu_lib_correctness` to compare the output of the
cuDNN and Tiramisu. Program outputs the maximum difference in the output for
each sequence. Note that small differences due to floating point arithmetic
in the first sequence grows exponentially as network progresses.

`configuration.h` sets the network parameters. There's also an option to switch
to double precision.
