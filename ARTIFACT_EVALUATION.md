## General

- Tiramisu website: http://tiramisu-compiler.github.io/
- Tiramisu github repository: https://github.com/Tiramisu-Compiler/tiramisu

- You can install Tiramisu from sources or use the Tiramisu virtual machine.  Both procedures are described in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md

- Instructions for installing Tiramisu are provided in section "Building Tiramisu" in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md).  If there is a need, more detailed instructions are provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/INSTALL.md

## Try Tiramisu
- You can compile Tiramisu (without recompiling all of its submodules) as follows

      cd /home/b/tiramisu
      cd build
      make clean
      make tiramisu

- You can try a given benchmark X using

      make run_benchmark_X

for example, you can try warp_affine using

      make run_benchmark_warp_affine

This will generate code from the Tiramisu code and run it.  The performance numbers are difficult to produce on the virtual machine though.

Code for this benchmarks is provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/halide/warp_affine_tiramisu.cpp
