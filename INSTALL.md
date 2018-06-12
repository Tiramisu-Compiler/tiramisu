## Building Tiramisu (Short Version)

This section provides a short description of how to build Tiramisu.  A more detailed description is provided below.  The installation instructions below have been tested on Linux Ubuntu (14.04) and MacOS (10.12) but should work on other Linux and MacOS versions.

#### Prerequisites
###### Required
1) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).
2) [CMake](https://cmake.org/): version 3.5 or greater.
  
###### Optional
1) [OpenMPI](https://www.open-mpi.org/) and [OpenSSh](https://www.openssh.com/): to run the generated distributed code (MPI).
2) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): to run the generated CUDA code.
3) [Doxygen](http://www.stack.nl/~dimitri/doxygen/): to generate documentation.
4) [Libpng](http://www.libpng.org/pub/png/libpng.html) and [libjpeg](http://libjpeg.sourceforge.net/): to run Halide benchmarks.
5) [Intel MKL](https://software.intel.com/mkl): to run BLAS and DNN benchmarks.


#### Building
1) Get Tiramisu

        git clone https://github.com/Tiramisu-Compiler/tiramisu.git
        cd tiramisu

2) Get and install Tiramisu submodules (ISL, LLVM and Halide)

        ./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_DIR>

3) Optional: configure the tiramisu build by editing `configure.cmake`.  Needed only if you want to generate MPI or GPU code, or if you want to run the BLAS benchmarks.  A description of what each variable is and how it should be set is provided in comments in `configure.cmake`.

    - To use the GPU backend, set `USE_GPU` to `true`.  If the CUDA library is not found automatically while building Tiramisu, the user will be prompt to provide the path to the CUDA library.
    - To use the distributed backend, set `USE_MPI` to `true`.  If the MPI library is not found automatically, set the following variables: MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS.
    - Set MKL_PREFIX to run the BLAS benchmarks.

4) Build the main Tiramisu library

        mkdir build
        cd build
        cmake ..
        make -j tiramisu


## Building Tiramisu (Long Version)
#### Prerequisites

For MacOs, we provide instructions on how to install the missing packages using [Homebrew](https://brew.sh/) but you can install these packages using any other way.  If you do not have Homebrew, you can install it as described [here](https://docs.brew.sh/Installation).

###### Required

1) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).
        
        # On Ubuntu
        sudo apt-get install autoconf libtool
        
        # On MacOS
        sudo brew install autoconf libtool

2) [CMake](https://cmake.org/): version 3.5 or greater.
  
        # On Ubuntu
        sudo apt-get install cmake

        # On MacOS
        sudo brew install cmake


###### Optional Packages
1) [OpenMPI](https://www.open-mpi.org/) and [OpenSSh](https://www.openssh.com/): to run the generated distributed code (MPI).

        # On Ubuntu
        sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev libopenmpi-dbg
        sudo apt-get install openssh-client openssh-server

        # On MacOs
        sudo brew install open-mpi
        sudo brew install openssh

2) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): to run the generated CUDA code.

        # Downalod and install the CUDA toolkit
        https://developer.nvidia.com/cuda-toolkit


3) [Doxygen](http://www.stack.nl/~dimitri/doxygen/): to generate documentation.

        # On Ubuntu
        sudo apt-get install doxygen

        # On MacOs
        sudo brew install doxygen


4) [Libpng](http://www.libpng.org/pub/png/libpng.html) and [libjpeg](http://libjpeg.sourceforge.net/): to run Halide benchmarks.

        # On Ubuntu
        sudo apt-get install libpng-dev libjpeg-dev
        
        # On MacOS
        sudo brew install libpng libjpeg

5) [Intel MKL](https://software.intel.com/mkl): to run BLAS and DNN benchmarks.

        # Download and install Intel MKL from:
        https://software.intel.com/mkl


Tiramisu requires the following packages to be installed: ISL, Halide and LLVM.  The user is supposed to install them by running the script `./utils/scripts/install_submodules.sh`.  If the script fails, the user can still install them manually as described below.


##### Building ISL

Install the [ISL](http://repo.or.cz/w/isl.git) Library as follows

        cd 3rdParty/isl
        git submodule update --init --remote --recursive
        mkdir build/
        ./autogen.sh
        ./configure --prefix=$PWD/build/ --with-int=imath
        make -j
        make install

After installing ISL, you need to update the following paths in `configure.cmake` to point to the ISL prefix (include and lib directories)

    ISL_INCLUDE_DIRECTORY: path to the ISL include directory
    ISL_LIB_DIRECTORY: path to the ISL library (lib/)

If the above fails, check the ISL [README](http://repo.or.cz/isl.git/blob/HEAD:/README) file for details on how you should install it in general.  Make sure that autoconf and libtool are installed on your system before building ISL.

###### Building LLVM

LLVM-5.0 or greater: required by the Halide framework. Check the section "Acquiring LLVM" in the Halide [README](https://github.com/halide/Halide/blob/master/README.md) for details on how to get LLVM, build and install it.


##### Building Halide

You need first to set the variable `LLVM_CONFIG_BIN` to point to the folder that contains `llvm-config` in the installed LLVM.  You can set this variable in `configure.cmake`. If you installed LLVM from source in a directory called `build/`, the path is usually set as follows

    LLVM_CONFIG_BIN=<path to llvm>/build/bin/

If you installed LLVM from the distribution packages, you need to find where it was installed and make `LLVM_CONFIG_BIN` point to the folder that contains `llvm-config`.

To get the Halide submodule and compile it run the following commands (in the Tiramisu root directory)

    git submodule update --init --remote
    cd Halide
    git checkout tiramisu
    make -j

You may get an access rights error from git when running trying to retrieve Halide. To solve this error, be sure to have your machine's ssh key added to your github account, the steps to do so could be found [HERE](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).

##### Building Tiramisu

To build Tiramisu, assuming you are in the Tiramisu root directory

        mkdir build
        cd build
        cmake ..
        make -j tiramisu

You need to add the Halide library path to your system library path.

    # On Ubuntu
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TIRAMISU_ROOT_DIRECTORY>/3rdParty/Halide/lib/

    # On MacOs
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<TIRAMISU_ROOT_DIRECTORY>/3rdParty/Halide/lib/
