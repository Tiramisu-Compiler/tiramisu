#! /bin/bash

# Get, compile and install Halide
# If this tep fails, please do it manually by following the
# steps indicated in https://github.com/halide/Halide
git submodule update --init --remote
cd Halide
git checkout tiramisu
make -j
