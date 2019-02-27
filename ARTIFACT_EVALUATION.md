## General

- Tiramisu website: http://tiramisu-compiler.github.io/
- Tiramisu github repository: https://github.com/Tiramisu-Compiler/tiramisu

- You can install Tiramisu from sources or use the Tiramisu virtual machine.
  * Instructions for using the virtual machine are provided in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md) in the section "Tiramisu on a Virtual Machine". If you use the Virtual machine you do not need to compile Tiramisu. It is already compiled and ready to use.
  * Instructions for installing Tiramisu from sources are also provided in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md) in the section "Building Tiramisu".  If there is a need, more detailed instructions are provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/INSTALL.md

## Trying Tiramisu

#### Tutorials
- A set of introductory tutorials is provided here: https://github.com/Tiramisu-Compiler/tiramisu/tree/master/tutorials 

- Assuming you are in the "build" directory, and assuming you have Tiramisu compiled, you can try a given tutorial X using

      make run_developers_tutorial_X

For example you can try tutorial_01 using

      cd <path-to-tiramisu>/build
      make run_developers_tutorial_01

- The code for Tutorial 01 is provided in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/tutorials/developers/tutorial_01/tutorial_01.cpp

You can explore the code of the other tutorials (up to tutorial 4C) and compile/run them.

#### Running Tests
- You can run the Tiramisu tests as follows

      cd <path-to-tiramisu>/build
      ctest

More details about running the tests are provided in the Tiramisu [README file](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md) in section "Run Tests".

#### Running the benchmarks

- You can try a given benchmark X using

      make run_benchmark_X

For example, you can try warp_affine using

      cd <path-to-tiramisu>/build
      make run_benchmark_warp_affine

This will generate code from the Tiramisu warp_affine benchmark and will run it.  Code for this benchmarks is available in https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/halide/warp_affine_tiramisu.cpp

- To try the cvtcolor benchmark

      make run_benchmark_cvt_color

If you compile Tiramisu yourself, you need to make sure that libpng and libjpeg are installed and enabled in the configure.cmake file. Please refer to the [Benchmarks README](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/benchmarks/README.md) file for more details.
