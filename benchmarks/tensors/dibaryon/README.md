## Quick Guide
* [Build Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu#building-tiramisu-from-sources).
    * If your machine supports avx2 and fma (fused multipliy add) instructions, you can enable those two fitures in Tiramisu by uncommenting [these two lines](https://github.com/Tiramisu-Compiler/tiramisu/blob/85fe07e465790b1254606079b3060db5af7fb36a/src/tiramisu_codegen_halide.cpp#L3928) before building Tiramisu.

* Generate the object files for the accelerated functions:

```
    cd benchmarks/
    ./compile_and_run_benchmarks.sh tensors/dibaryon/tiramisu_make_local_single_double_block/ tiramisu_make_local_single_double_block 
```

The previous script will go to the folder "tensors/dibaryon/tiramisu_make_local_single_double_block/" and build the Tiramisu code "tiramisu_make_local_single_double_block" and run it to generate the object file.


* Edit the file [qblocks_2pt_parameters.h](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/tensors/dibaryon/reference/qblocks_2pt_parameters.h) to specify the sizes that you want. This file is used by both the Tiramisu code and by the reference code.


* Make the reference code (no Tiramisu)

```
    cd reference
    make
```

* Make the Tiramisu version of the code

```
    cd reference
    make tiramisu
```

* In order to run code compiler with Tiramisu, you need to add the following library paths:

```
    ${TIRAMISU_ROOT}/3rdParty/Halide/bin/
    ${TIRAMISU_ROOT}/build
```

where ${TIRAMISU_ROOT} is the root directory of Tiramisu source (i.e., top directory). For example, on Linux you can do

```
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TIRAMISU_ROOT}/3rdParty/Halide/bin/:${TIRAMISU_ROOT}/build/ ./qblocks_2pt_test
```
