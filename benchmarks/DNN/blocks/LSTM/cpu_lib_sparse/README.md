Basic CPU implementation of an LSTM Sparse forward pass.
Prerequisites :
    You need to turn on MKL wrappers and give the path to MKL by setting MKL_PREFIX, these can be done in the tiramisu/configure.cmake file.

Run the benchmark :

  From the build directory tiramisu/build/benchmarks/DNN/blocks/LSTM/cpu_lib_sparse/ execute:
    make

  - wrapper_lstm_sparse file is generated in tiramisu/benchmarks/DNN/blocks/LSTM/cpu_lib_sparse/

  To compare to MKLDNN, run ./compile_and_run_mkldnn.sh, then run ./wrapper_lstm_sparse
  To compare to MKL Sparse, run ./compile_and_run_mkl_sparse.sh intel64 then run ./wrapper_lstm_sparse (here intel64 can be something else depending on the architecture) 
