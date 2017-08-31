#! /bin/bash

# Get, compile and install ISL
# If this step fail please do it manually by following the steps
# indicated in http://repo.or.cz/isl.git/blob/HEAD:/README
# Make sure you have all the ncessary dependencies installed.
# Please install isl in  isl/build/ using the --prefix argument
# for configure as shown below.
cd 3rdParty/gmp-6.1.2/
if [ ! -d "build" ]; then
	mkdir build/
fi
./configure --prefix=$PWD/build/
make -j
make install

cd ../isl
git submodule update --init --remote
if [ ! -d "build" ]; then
	mkdir build/
fi
./autogen.sh
./configure --prefix=$PWD/build/ --with-gmp-prefix=$PWD/../gmp-6.1.2/build/ 
make -j
make install
cd ..
