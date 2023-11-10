#!/bin/bash

USE_LIBJPEG=0
USE_LIBPNG=0

if [ "$#" -eq 0 ]; then
	echo "Usage: install_submodules.sh <TIRAMISU_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=`realpath ${1}`
CMAKE=cmake
CORES=4

# For Travis build we skip LLVM installation and use a custom binary.
# Second argument specifies the custom path of the LLVM bin dir.
if [ "$2" = "" ]; then
    LLVM_BIN_DIR=${PROJECT_SRC_DIR}/3rdParty/llvm/build/bin
else
    LLVM_BIN_DIR="$2"
fi

set -e
. ${PROJECT_SRC_DIR}/utils/scripts/functions.sh

# Install ISL into 3rdParty and Halide into the root of the tiramisu directory

# POSSIBLE ERRORS
# 1. If you get a permissions error when trying to clone these submodules, you may not have your ssh keys set up correctly
# in github. We use ssh to clone the repos, not https.
#
# 2. If you get an error that some file in a repo was not found (such as autogen.sh in isl), you may have partially
# cloned a submodule before but not completed it, meaning some of the files are missing. Make sure to delete the .git
# folder in that submodule directory to force the clone to happen again.

echo ${PROJECT_SRC_DIR}

# echo "#### Cloning submodules ####"

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}"
echo_and_run_cmd "git submodule update --init --recursive"


# Get ISL installed
echo "#### Installing isl ####"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/isl"
if [ ! -d "build" ]; then
    echo_and_run_cmd "mkdir build/"
fi
#echo_and_run_cmd "touch aclocal.m4 Makefile.am Makefile.in"
echo_and_run_cmd "./autogen.sh"
echo_and_run_cmd "./configure --prefix=$PWD/build/ --with-int=imath"
echo_and_run_cmd "make -j $CORES"
echo_and_run_cmd "make install"
echo "Done installing isl"




# Get LLVM installed
if [ "$2" = "" ]; then
    echo "#### Installing LLVM ####"
    echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/llvm"
    if [ ! -d "build" ]; then
        echo_and_run_cmd "mkdir build/"
    fi
    if [ ! -d "prefix" ]; then
        echo_and_run_cmd "mkdir prefix/"
    fi
    echo_and_run_cmd "cd build"
    echo_and_run_cmd "$CMAKE -G Ninja -S ../llvm -DHAVE_LIBEDIT=0 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_PROJECTS='clang;lld;clang-tools-extra' -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF -DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64;Mips;NVPTX;PowerPC' -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../prefix/ -DCMAKE_MAKE_PROGRAM='ninja' -DCMAKE_CXX_COMPILER='g++'"
    echo_and_run_cmd "cmake --build . -j $CORES"
    echo_and_run_cmd "cmake --install ."
    echo "### Done Installing LLVM###"
else
    echo "#### Skipping LLVM Installation ####"
fi
#    echo_and_run_cmd "$CMAKE -G Ninja -S ../llvm -DHAVE_LIBEDIT=0 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_PROJECTS='clang;lld;clang-tools-extra' -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF -DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64;Mips;NVPTX;PowerPC' -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../prefix/ -DCMAKE_C_COMPILER='mpicc' CMAKE_CXX_COMPILER='mpicxx' -DCMAKE_MAKE_PROGRAM='ninja'"

export CXX=""

# # Get halide installed
# echo "#### Installing Halide ####"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/Halide"
if [ "${USE_LIBJPEG}" = "0" ]; then
    CXXFLAGS_JPEG="-DHALIDE_NO_JPEG=1"
fi
if [ "${USE_LIBPNG}" = "0" ]; then
    CXXFLAGS_PNG="-DHALIDE_NO_PNG=1"
fi
echo_and_run_cmd "mkdir -p ${PROJECT_SRC_DIR}/3rdParty/Halide/install"
echo_and_run_cmd "cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=${PROJECT_SRC_DIR}/3rdParty/llvm/prefix/lib/cmake/llvm -DCMAKE_MAKE_PROGRAM='ninja' -DCMAKE_CXX_COMPILER='g++' -DCMAKE_CXX_FLAGS='-std=c++17' -DCMAKE_INSTALL_PREFIX=${PROJECT_SRC_DIR}/3rdParty/Halide/install ${CXXFLAGS_JPEG} ${CXXFLAGS_PNG} -S . -B build"
echo_and_run_cmd "cmake --build build -j ${CORES}"
echo_and_run_cmd "cmake --install build"

cd ${PROJECT_SRC_DIR}
echo "Done installing Halide"
echo "Having installed all depends, we suggest you set your PATH and LD_LIBRARY_PATH as follows:"
echo "export PATH=${PROJECT_SRC_DIR}/3rdParty/llvm/build/bin:$PATH"
echo "export LD_LIBRARY_PATH=${PROJECT_SRC_DIR}/3rdParty/Halide/build/src:${PROJECT_SRC_DIR}/3rdParty/llvm/build/lib:$LD_LIBRARY_PATH"

