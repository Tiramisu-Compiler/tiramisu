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
