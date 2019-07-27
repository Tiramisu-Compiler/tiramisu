The files in this folder are organized as follows:

* General
  - clean.sh : remove some useless files.
  - configure.h: define size of input matrices.

* Tiramisu
  - conv_layer_generator.cpp: Tiramisu code generator.

* Intel MKL
  - compile_mkldnn_and_run.sh : compile Intel MKL DNN code.
  - s_score_sample.c: code that calls Intel MKL DNN convolution. We copied this code from the Intel MKL sample files.
