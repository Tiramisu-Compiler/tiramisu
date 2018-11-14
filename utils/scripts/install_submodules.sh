#!/bin/bash

USE_LIBJPEG=0

if [ "$#" -eq 0 ]; then
	echo "Usage: install_submodules.sh <TIRAMISU_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=$1
CMAKE=cmake
CORES=1
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

echo "#### Cloning submodules ####"

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}"
echo_and_run_cmd "git submodule update --init --remote --recursive"


# Get ISL installed
echo "#### Installing isl ####"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/isl"
if [ ! -d "build" ]; then
    echo_and_run_cmd "mkdir build/"
fi
echo_and_run_cmd "touch aclocal.m4 Makefile.am Makefile.in"
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
    echo_and_run_cmd "$CMAKE -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD='X86;ARM;AArch64;Mips;NVPTX;PowerPC' -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_INSTALL_PREFIX=$PWD/../prefix/ -DLLVM_EXTERNAL_CLANG_SOURCE_DIR=${PROJECT_SRC_DIR}/3rdParty/clang"
    echo_and_run_cmd "make -j $CORES"
    echo_and_run_cmd "make install"
else
    echo "#### Skipping LLVM Installation ####"
fi



# Set LLVM_CONFIG and CLANG env variables
export CLANG=${LLVM_BIN_DIR}/clang
export LLVM_CONFIG=${LLVM_BIN_DIR}/llvm-config



# Get halide installed
echo "#### Installing Halide ####"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/Halide"
echo_and_run_cmd "git checkout tiramisu_64_bit"
echo_and_run_cmd "git pull"
if [ "${USE_LIBJPEG}" = "0" ]; then
    echo_and_run_cmd "make CXXFLAGS=\"-DHALIDE_NO_JPEG\" -j $CORES"
else
    echo_and_run_cmd "make clean"
    echo_and_run_cmd "make -j $CORES"
fi



cd ${PROJECT_SRC_DIR}
echo "Done installing Halide"
