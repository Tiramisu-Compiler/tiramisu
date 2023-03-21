# Single baryon blocks correlator GPU benchmark

To run this benchmark make sure you have USE_GPU option enabled in configure.cmake file (or from the cmake command line using `cmake <PATH_TO_TIRAMISU> -DUSE_GPU=1 <YOUR_BUILD_DIRECTORY>`) and go to the build directory and run 

    `cmake --build . --target run_gpu_single_tiramisu_make_fused_identical_baryon_blocks_correlator`
