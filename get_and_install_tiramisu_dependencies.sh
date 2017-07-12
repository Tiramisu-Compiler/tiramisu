#! /bin/bash

# Step 1: Get, compile and install ISL
# If this step fail please do it manually by following the steps
# indicated in http://repo.or.cz/isl.git/blob/HEAD:/README
# Make sure you have all the ncessary dependencies installed.
# Please install isl in  isl/build/ using the --prefix argument
# for configure as shown below.
git clone git://repo.or.cz/isl.git
cd isl
mkdir build/
./autogen.sh
./configure --prefix=$PWD/build/
make -j
make install
cd ..

# Step 2: Get, compile and install Halide
# If this tep fails, please do it manually by following the
# steps indicated in https://github.com/halide/Halide
git submodule update --init --remote
cd Halide
git checkout tiramisu
make -j

# Step 3: Build Tiramisu as indicated in the Tiramisu README file.
# If you did the steps above manually please make sure that
# the variables ISL_INCLUDE_DIRECTORY, ISL_LIB_DIRECTORY,
# HALIDE_SOURCE_DIRECTORY and HALIDE_LIB_DIRECTOR are all
# set correctly in the Tiramisu makefile (which you can find
# in the Tiramisu root directory).
