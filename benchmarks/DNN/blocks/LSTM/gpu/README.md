LSTM implementation in Tiramisu. The network is the same as in Nvidia blogpost:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/

Unlike [LSTM/gpu_lib](../gpu_lib) benchmark, this doesn't use external libraries for matrix multiplication.

To run the benchmark you need to have cuDNN installed on the machine.
Set `USE_GPU` and `USE_CUDNN` to true in `configure.cmake`, and set
`CUDNN_LOCATION` to cuDNN location.

From Tiramisu build directory:
- `make run_benchmark_lstm_gpu` to run the benchmark
