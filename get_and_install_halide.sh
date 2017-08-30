#! /bin/bash

source configure_paths.sh

export CLANG=${LLVM_CONFIG_BIN}clang
export LLVM_CONFIG=${LLVM_CONFIG_BIN}llvm-config

# Get, compile and install Halide
# If this tep fails, please do it manually by following the
# steps indicated in https://github.com/halide/Halide
git submodule update --init --remote
cd Halide
git checkout tiramisu
git pull
make -j
