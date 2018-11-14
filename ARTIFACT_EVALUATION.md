## General

- Tiramisu website: http://tiramisu-compiler.github.io/
- Tiramisu github repository: https://github.com/Tiramisu-Compiler/tiramisu

- You can install Tiramisu from sources or use the Tiramisu virtual machine.  Both procedures are described in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md

- Instructions for installing Tiramisu are provided in section "Building Tiramisu" in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md).  If there is a need, more detailed instructions are provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/INSTALL.md

## Trying Tiramisu
- You can compile Tiramisu (without recompiling all of its submodules) as follows

      cd /home/b/tiramisu
      cd build
      make clean
      make tiramisu

- A set of introductory tutorials is provided here: https://github.com/Tiramisu-Compiler/tiramisu/tree/master/tutorials 

- You can try a given tutorial X using

      make run_developers_tutorial_X

for example, you can try tutorial_01 using

      make run_developers_tutorial_01

- You can try a given benchmark X using

      make run_benchmark_X

for example, you can try warp_affine using

      make run_benchmark_warp_affine

This will generate code from the Tiramisu warp_affine benchmark and will run it.  Code for this benchmarks is available in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/halide/warp_affine_tiramisu.cpp

- To try the cvtcolor benchmark

      make run_benchmark_cvt_color
