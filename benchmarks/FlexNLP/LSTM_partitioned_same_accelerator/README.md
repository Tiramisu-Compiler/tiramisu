# LSTM split on a single accelerator
# Files
  - configure.h : contains different parameters such as HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE
  - flexnl_tiramisu_generator.cpp : tiramisu code that uses the FlexNLP tiramisu functions to run an LSTM
  - flexlp_tiramisu_wrapper.cpp : wrapper that declares the buffers then uses the generated (compiled) Tiramisu function
  - clean.sh : cleans object files and executable files
  - CMakeLists.txt : for compiling the code

# How to Run
To run this code, go to tiramisu/build/benchmarks/FlexNLP/LSTM_partitioned_same_accelerator and execute :

> make

The wrapper_flexnlp_tiramisu_lstm file will be generated in tiramisu/benchmarks/FlexNLP/LSTM_partitioned_same_accelerator.

Then go back to tiramisu/benchmarks/FlexNLP/LSTM_partitioned_same_accelerator and run

> ./wrapper_flexnlp_tiramisu_lstm_partitioned

The message "Finished" will  be prompted if it executes successfully (Note that the function doesn't consider bias for now as we agreed on not using it for now)
