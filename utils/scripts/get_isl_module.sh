#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: get_isl_module.sh <TIRAMISU_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=$1
VERSION=23

set -e
. ${PROJECT_SRC_DIR}/utils/scripts/functions.sh

echo "### Remove old ISL, get a new one and decompress it. ####"

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/"
echo_and_run_cmd "rm -rf isl/"
echo_and_run_cmd "wget https://libisl.sourceforge.io/isl-0.$VERSION.tar.gz"
echo_and_run_cmd "tar -xvf isl-0.$VERSION.tar.gz"
echo_and_run_cmd "rm -rf isl-0.$VERSION.tar.gz"
echo_and_run_cmd "mv isl-0.$VERSION isl/"
echo_and_run_cmd "cd ./isl"
echo_and_run_cmd "mkdir build"
echo_and_run_cmd "./configure --prefix=${PROJECT_SRC_DIR}/3rdParty/isl/build --enable-static --with-pic"
echo_and_run_cmd "make"
echo_and_run_cmd "make install"
