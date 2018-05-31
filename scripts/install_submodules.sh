#!/bin/bash

PROJECT_SRC_DIR=$1
LLVM_CONFIG_BIN_DIR=$2

set -e
. ${PROJECT_SRC_DIR}/scripts/functions.sh

# Install ISL into 3rdParty and Halide into the root of the tiramisu directory

# POSSIBLE ERRORS
# 1. If you get a permissions error when trying to clone these submodules, you may not have your ssh keys set up correctly
# in github. We use ssh to clone the repos, not https.
#
# 2. If you get an error that some file in a repo was not found (such as autogen.sh in isl), you may have partially
# cloned a submodule before but not completed it, meaning some of the files are missing. Make sure to delete the .git
# folder in that submodule directory to force the clone to happen again.



echo ${PROJECT_SRC_DIR}
echo ${LLVM_CONFIG_BIN_DIR}

echo "#### Cloning submodules ####"

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}"
echo_and_run_cmd "git submodule update --init --remote --recursive"

# Get isl installed
echo "#### Installing isl ####"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/isl"
if [ ! -d "build" ]; then
    echo_and_run_cmd "mkdir build/"
fi
echo_and_run_cmd "./autogen.sh"
echo_and_run_cmd "./configure --prefix=$PWD/build/ --with-int=imath"
echo_and_run_cmd "make -j"
echo_and_run_cmd "make install"

# Get halide installed
echo "#### Installing Halide ####"
echo_and_run_cmd "export CLANG=${LLVM_CONFIG_BIN_DIR}/clang"
echo_and_run_cmd "export LLVM_CONFIG=${LLVM_CONFIG_BIN_DIR}/llvm-config"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/Halide"
echo_and_run_cmd "git checkout tiramisu_64_bit"
echo_and_run_cmd "git pull"
echo_and_run_cmd "make -j"