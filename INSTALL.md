## Building Tiramisu (Long Version)

Tiramisu has been well tested on Linux Ubuntu (18.04) and MacOS (10.12), if you find any problem installing Tiramisu please consider using one of these systems.

#### Prerequisites

For MacOs, we provide instructions on how to install the missing packages using [Homebrew](https://brew.sh/) but you can install these packages using any other way.  If you do not have Homebrew, you can install it as described [here](https://docs.brew.sh/Installation).

If you are installing Tiramisu on Ubuntu 18.04, you can install all the required packages for generating CPU code by running the following command

      sudo apt-get install autoconf libtool git cmake gcc g++ libpng-dev zlib1g-dev libjpeg-dev

The next section provides more details about the installation of these prerequisites.

###### Required

1) [CMake](https://cmake.org/): version 3.5 or greater.
  
        # On Ubuntu
        sudo apt-get install cmake

        # On MacOS
        sudo brew install cmake

2) [Autoconf](https://www.gnu.org/software/autoconf/) and [libtool](https://www.gnu.org/software/libtool/).

        # On Ubuntu
        sudo apt-get install autoconf libtool

        # On MacOS
        sudo brew install autoconf libtool

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


Tiramisu requires the following packages to be installed: ISL, Halide and LLVM.  The user is supposed to install them by running the script `./utils/scripts/install_submodules.sh <TIRAMISU_ROOT_PATH>`.  If the script fails, the user can still install them manually as described below.

##### Building ISL

Install the [ISL](http://isl.gforge.inria.fr/) Library as follows

        cd 3rdParty/isl
        mkdir build/
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

Halide requires also libdl and libpthread. Both libraries should be available on every Linux and MacOs system and do not installation usually but you might need to install them if you get an error such as "-ldl not found" or "-lpthread not found".

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
