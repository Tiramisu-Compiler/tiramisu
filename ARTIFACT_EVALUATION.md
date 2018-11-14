## General

- Tiramisu website: http://tiramisu-compiler.github.io/
- Tiramisu github repository: https://github.com/Tiramisu-Compiler/tiramisu

- The instructions for installing Tiramisu from sources are provided in section "Building Tiramisu" in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md).  If there is a need, more detailed instructions are provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/INSTALL.md

- A virtual machine that has Tiramisu pre-built will be added soon (20GB being uploaded to the servers).

## Trying Tiramisu
- You can compile Tiramisu (without recompiling all of its submodules) as follows

      cd /home/b/tiramisu
      cd build
      make clean
      make tiramisu

- A set of introductory tutorials is provided here: https://github.com/Tiramisu-Compiler/tiramisu/tree/master/tutorials 

- Assuming you are in the "build" directory, you can try a given tutorial X using

      make run_developers_tutorial_X

for example, you can try tutorial_01 using

      make run_developers_tutorial_01

The code for Tutorial 01 is provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/tutorials/developers/tutorial_01/tutorial_01.cpp

- You can try a given benchmark X using

      make run_benchmark_X

for example, you can try warp_affine using

      make run_benchmark_warp_affine

This will generate code from the Tiramisu warp_affine benchmark and will run it.  Code for this benchmarks is available in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/halide/warp_affine_tiramisu.cpp

- To try the cvtcolor benchmark

      make run_benchmark_cvt_color
