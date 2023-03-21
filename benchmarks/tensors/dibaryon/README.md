## CPU Code
* [Build Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu#building-tiramisu-from-sources).
    * If your machine supports avx2 and fma (fused multipliy add) instructions, you can enable those two fitures in Tiramisu by uncommenting [these two lines](https://github.com/Tiramisu-Compiler/tiramisu/blob/85fe07e465790b1254606079b3060db5af7fb36a/src/tiramisu_codegen_halide.cpp#L3928) before building Tiramisu.

* Generate the object files for the accelerated functions:

```
    cd benchmarks/
    ./compile_and_run_benchmarks.sh tensors/dibaryon/tiramisu_make_fused_dibaryon_blocks_correlator/ tiramisu_make_fused_dibaryon_blocks_correlator 
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
    make run
```


## GPU Dibaryon Code

### Building and running the benchmark:
* [Build Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu#building-tiramisu-from-sources) and make sure to enable the USE_GPU parameter to use GPU backend.
* To run the benchmark use the following command at the build directory:

    make  run_gpu_tiramisu_make_fused_dibaryon_blocks_correlator

### Benchmark paramters:
* The parameters for the gpu dibaryon benchmark are located at [qblocks_2pt_parameters.h](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/tensors/dibaryon/gpu_tiramisu_make_fused_dibaryon_blocks_correlator/qblocks_2pt_parameters.h)
* The most important parameters are:
    * P_Vsrc and P_Vsnk: the size of the benchmark
    * P_sites_per_rank and P_src_sites_per_rank:
        for the GPU benchmark P_sites_per_rank and P_src_sites_per_rank indicates the number of GPU threads used by the some parts of the dibaryon code whereas P_Vsnk / P_sites_per_rank and P_Vsrc / P_src_sites_per_rank indicate the number of GPU blocks.
        Both parameters need to divide P_Vsrc and P_src_sites_per_rank and be different than 1.
        To make the code more efficient make sure to make P_sites_per_rank and P_src_sites_per_rank as high as possible while making P_Vsnk / P_sites_per_rank and P_Vsrc / P_src_sites_per_rank closer to the number of SMs in the GPU.
        For example for the Nvidia Tesla V100 GPU has 80 multiprocessor units and RTX2080 has 46
        If P_Vsrc = 512 and the used GPU is V100, for best performance we can use P_sites_per_rank = 512 / 4 = 128
    * P_tiling_factor:
        Due to memory limitations, the GPU code needs to divide the computations into tiles.
        This parameter needs to be as low as possible while keeping the used GPU memory below that of the GPU: for example for the V100 GPU we needed to keep P_Vsrc / P_tiling_factor less than or equal to 64

### Another Method

To run a benchmark make sure you have USE_GPU option enabled in configure.cmake file (or from the cmake command line using `cmake <PATH_TO_TIRAMISU> -DUSE_GPU=1 <YOUR_BUILD_DIRECTORY>`) and go to the build directory and run 

    `cmake --build . --target run_gpu_tiramisu_make_fused_baryon_blocks_correlator`


