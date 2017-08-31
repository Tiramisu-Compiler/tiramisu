#! /bin/bash

# Get, compile and install ISL
# If this step fail please do it manually by following the steps
# indicated in http://repo.or.cz/isl.git/blob/HEAD:/README
# Make sure you have all the ncessary dependencies installed.
# Please install isl in  isl/build/ using the --prefix argument
# for configure as shown below.

cd 3rdParty/isl
git submodule update --init --remote --recursive

if [ ! -d "build" ]; then
	mkdir build/
fi
./autogen.sh
./configure --prefix=$PWD/build/ --with-int=imath
make -j
make install
cd ..
