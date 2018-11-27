Basic CPU implementation of an LSTM forward pass. From the build directory:
- `make run_benchmark_lstm_cpu` to run the LSTM
It takes about 50s to finish the inference.
The model is the same as in: https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/

TODO:
- Add loop optimizations
- Add correctness check
- Add timing and comparison with Halide and other implementations
