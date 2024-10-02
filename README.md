[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
<!--- [![Build Status](https://travis-ci.org/Tiramisu-Compiler/tiramisu.svg?branch=master)](https://travis-ci.org/Tiramisu-Compiler/tiramisu)--->

## Overview

Tiramisu is a compiler for expressing fast and portable data parallel computations.  It provides a simple C++ API for expressing algorithms (`Tiramisu expressions`) and how these algorithms should be optimized by the compiler.  Tiramisu can be used in areas such as linear and tensor algebra, deep learning, image processing, stencil computations and machine learning.

The Tiramisu compiler is based on the polyhedral model thus it can express a large set of loop optimizations and data layout transformations.  Currently it targets (1) multicore X86 CPUs, (2) Nvidia GPUs, (3) Xilinx FPGAs (Vivado HLS) and (4) distributed machines (using MPI).  It is designed to enable easy integration of code generators for new architectures.

### Example

The following is an example of a Tiramisu program specified using the C++ API.

```cpp
// C++ code with a Tiramisu expression.
#include "tiramisu/tiramisu.h"
using namespace tiramisu;

void generate_code()
{
    // Specify the name of the function that you want to create.
    tiramisu::init("foo");

    // Declare two iterator variables (i and j) such that 0<=i<100 and 0<=j<100.
    var i("i", 0, 100), j("j", 0, 100);

    // Declare a Tiramisu expression (algorithm) that is equivalent to the following C code
    // for (i=0; i<100; i++)
    //   for (j=0; j<100; j++)
    //     C(i,j) = 0;
    computation C({i,j}, 0);

    // Specify optimizations
    C.parallelize(i);
    C.vectorize(j, 4);

    buffer b_C("b_C", {100, 100}, p_int32, a_output);
    C.store_in(&b_C);

    // Generate code
    C.codegen({&b_C}, "generated_code.o");
}
```




## Building Tiramisu from Sources

This section provides a description of how to build Tiramisu.  The installation instructions below have been tested on Linux Ubuntu (18.04) and MacOS (13.0.1) but should work on other Linux and MacOS versions.

### Prerequisites
###### Required
1) [CMake](https://cmake.org/): version 3.22 or greater.

2) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).

3) [Ninja](https://ninja-build.org/).

###### Optional
1) [OpenMPI](https://www.open-mpi.org/) and [OpenSSh](https://www.openssh.com/): if you want to generate and run distributed code (MPI).
2) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): if you want to generate and run CUDA code.

3) Python 3.8 or higher if you want to use the python bindings. (Along with Pybind 2.10.2, Cython, and Numpy).


### Build Methods

There are 3 ways to build Tiramisu:
1) From [spack](https://packages.spack.io/package.html?name=tiramisu), which will build everything from source for you.
2) From source, but using system package managers for dependencies.
3) Purely from source with our install script.

The last two only differ only in how they setup the dependenies.

#### Method 1: Build from spack

Install spack and then run:
```bash
spack install tiramisu
```

#### Method 2: Build from source but install dependencies using system package managers

There are two steps:
1) Install the dependencies (either using Homebrew or using Apt).
2) Use Cmake to build Tiramisu.

##### Install the dependencies

###### Install the dependencies using Homebrew
If you are on MacOS and using Homebrew, you can run the following commands to setup the dependencies:
```bash
brew install cmake
brew install llvm@14
brew install halide
brew install isl
brew link halide
brew link isl
```

If any of these ask you to update your path, do so. For example, using the following command, you can find the isl include and library directories:
```bash
brew info isl
ISL_INCLUDE_DIRECTORY=..
ISL_LIB_DIRECTORY=..
```

###### Install the dependencies using Apt

If you are on Ubuntu/Debian, you can use apt to setup the dependencies:

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 14 all
sudo apt-get install liblld-14-dev llvm-14-runtime
sudo apt-get install libllvm14 llvm-14-dev
sudo apt-get install llvm14-*
sudo apt-get install halide
sudo apt-get install libisl-dev
```


 Using the following command, you can find the isl include and library directories:
```bash
dpkg -L libisl-dev
ISL_INCLUDE_DIRECTORY=..
ISL_LIB_DIRECTORY=..
```

##### Building Tiramisu with cmake

1) Get Tiramisu
```bash
git clone https://github.com/Tiramisu-Compiler/tiramisu.git
cd tiramisu
mkdir build
```

2) Setup the configure.cmake. In particular, choose if you want to use a GPU or MPI setup. Choose if you want to use the python bindings. Choose if you want to us the auto scheduler. You may need to add other options to support these.

3) Configure:

```bash
cmake . -B build -DISL_LIB_DIRECTORY=$ISL_LIB_DIRECTORY -DISL_INCLUDE_DIRECTORY=$ISL_INCLUDE_DIRECTORY -DPython3_EXECUTABLE=`which python3`
```

If you want to install, add `CMAKE_INSTALL_PREFIX`. If you are installing the python bindings, add `Tiramisu_INSTALL_PYTHONDIR` to tell Tiramisu where to place a python package. You will need add these install locations to the relevant path variables such as `PYTHONPATH` and `LD_LIBRARY_PATH`.

4) Build:
```bash
cmake --build build
```

You can also install if you want via `cmake --install`.


#### Method 3: Build from source, but install dependencies using our script

There are two steps:
1) Install the dependencies using our script.
2) Use Cmake to build Tiramisu.

##### Building Dependencies via Script
1) Get Tiramisu
```bash
git clone https://github.com/Tiramisu-Compiler/tiramisu.git
cd tiramisu
```
2) Get and install Tiramisu submodules (ISL, LLVM and Halide).  This step may take between few minutes to few hours (downloading and compiling LLVM is time consuming).
```bash
./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_DIR>
```
    - Note: Make sure `<TIRAMISU_ROOT_DIR>` is absolute path!

3) Optional: configure the tiramisu build by editing `configure.cmake`.  Needed only if you want to generate MPI or GPU code, run the BLAS benchmarks, or if you want to build the autoscheduler module.  A description of what each variable is and how it should be set is provided in comments in `configure.cmake`.

    - To use the GPU backend, set `USE_GPU` to `TRUE`. If the CUDA library is not found automatically while building Tiramisu, the user will be prompt to provide the path to the CUDA library.
    - To use the distributed backend, set `USE_MPI` to `TRUE`. If the MPI library is not found automatically, set the following variables: MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS.
    - To build the autoscheduler module, set `USE_AUTO_SCHEDULER` to `TRUE`.

4) Add Halide's cmake to the `CMAKE_PREFIX_PATH`:
```bash
export CMAKE_PREFIX_PATH=$TIRAMISU_ROOT_DIR/3rdParty/Halide/install/:$CMAKE_PREFIX_PATH
```
5) Build the main Tiramisu library
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

6) If you want to build the autoscheduler module, set `USE_AUTO_SCHEDULER` to `TRUE` in `configure.cmake`, and after building Tiramisu :
```bash
make tiramisu_auto_scheduler
```


#### Method 4: Installing Tiramisu with script (Linux only)
This section describes how to install Tiramisu on Linux distributions based on Debian, Ubuntu, Arch and Fedora.

Note that this is the fastest installation method, as it doesn't require building the large dependencies (LLVM and Halide) but downloads their binaries and links them directly.

##### 1. Clone the repo and checkout to `merge_attempt` branch
```bash
git clone https://github.com/Tiramisu-Compiler/tiramisu
cd tiramisu
git checkout merge_attempt
```

##### 2. Run the installation script
```bash
bash build-install.sh -o <installation_directory>
```

This script will automatically download the dependencies and build Tiramisu.


###### 2.1. Script arguments:
`-o <install_directory>`:
- **Description**: Specifies the directory where Tiramisu will be installed.
- **Default**: If not provided, the script will use the default install directory `$PWD/install`.
- **Usage**: `./build-install.sh -o /path/to/install/dir`
    - Example: `./build-install.sh -o /home/user/tiramisu_install`


###### 2.2. Script side effects:
The script saves the following environment variables to the user's `.bashrc` and `.zshrc` files:

- `TIRAMISU_ROOT`
	- **Description**: Specifies the root directory of the Tiramisu project (the current working directory when the script is executed).
	- **Value**: The current directory where the script is executed (`$PWD`).
	- **Purpose**: Used to reference the Tiramisu project directory for build and configuration purposes.

- `LD_LIBRARY_PATH`
	- **Description**: Specifies the directory paths where dynamic libraries are searched during execution.
	- **Value**: `${TIRAMISU_ROOT}/3rdParty/Halide-bin/lib:$LD_LIBRARY_PATH`
	- **Purpose**: Adds the path to Halide binaries to ensure that any executables linked against Halide can find the necessary libraries.

- `CMAKE_PREFIX_PATH`
	- **Description**: Provides the search paths for CMake to locate installed packages.
	- **Value**: `${TIRAMISU_ROOT}/3rdParty/Halide-bin/:$CMAKE_PREFIX_PATH`
	- **Purpose**: Tells CMake where to find the Halide binaries during the build process.

##### 2.3. Example of what will be added to `.bashrc` and `.zshrc`:

```bash
# Set up environment variables for Tiramisu
export TIRAMISU_ROOT=/path/to/tiramisu
export LD_LIBRARY_PATH=/path/to/tiramisu/3rdParty/Halide-bin/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/path/to/tiramisu/3rdParty/Halide-bin/:$CMAKE_PREFIX_PATH
```

These environment variables ensure that the Tiramisu project has access to the necessary Halide libraries and paths for successful builds and runtime execution.


## Old Tiramisu on a Virtual Machine
Users can use the Tiramisu [virtual machine disk image](http://groups.csail.mit.edu/commit/software/TiramisuVM.zip).  The image is created using virtual box (5.2.12) and has Tiramisu already pre-compiled and ready for use. It was compiled using the same instructions in this README file.

Once you download the image, unzip it and use virtual box to open the file 'TiramisuVM.vbox'.

Once the virtual machine has started, open a terminal, then go to the Tiramisu directory

    cd /home/b/tiramisu/

If asked for a username/password

    Username:b
    Password:b

## Getting Started
- Build [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu/).
- Read the [Tutorials](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/tutorials/README.md).
- Read the [Tiramisu Paper](https://arxiv.org/abs/1804.10694).
- Subscribe to Tiramisu [mailing list](https://lists.csail.mit.edu/mailman/listinfo/tiramisu).
- Read the compiler [internal documentation](https://tiramisu-compiler.github.io/doc/) (if you want to contribute to the compiler).


## Run Tests

To run all the tests, assuming you are in the build/ directory

    make test

or

    ctest

To run only one test (test_01 for example)

    ctest -R 01

This will compile and run the code generator and then the wrapper.

To view the output of a test pass the `--verbose` option to `ctest`.
