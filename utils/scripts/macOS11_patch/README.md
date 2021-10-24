# Installing Tiramisu on macOS 11 or newer

This section provides quick instructions for a basic installation of tiramisu on macOS >= 11
1) Make sure the following prerequisites are installed:
    - autoconf
    - libtool
    - cmake>=3.5 
    - gcc==7.5 

    Otherwise, you can install them via [Homebrew](https://brew.sh):
    ```sh
    brew install autoconf, libtool, cmake, gcc@7
    ```

2) Clone Tiramisu:
    ```sh
    git clone https://github.com/Tiramisu-Compiler/tiramisu.git
    cd tiramisu
    ```
3) Apply the macOS 11 installation patch:
    ```sh
    patch -p1 -i utils/scripts/macOS11_patch/installation_patch.patch
    ```
4) Export the following environment variables:
    ```sh
    export MACOSX_DEPLOYMENT_TARGET=10.16.3
    export CXX=g++-7
    export CC=gcc-7
    export TIRAMISU_ROOT=$(pwd)
    ```

5) Install Tiramisu and its submodules (ISL, LLVM, and Halide). This step may take some time to complete:
    ```sh
    ./utils/scripts/install_submodules.sh $TIRAMISU_ROOT
    mkdir build
    cd build
    cmake ..
    make -j tiramisu
    ```


For more details about the installation steps and the different building configurations, please refer to the main [Installation Guide](https://github.com/Tiramisu-Compiler/tiramisu/blob/master/README.md#building-tiramisu-from-sources).

