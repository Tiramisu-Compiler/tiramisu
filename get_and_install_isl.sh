#! /bin/bash

# Get, compile and install ISL
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
