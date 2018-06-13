#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: get_isl_module.sh <TIRAMISU_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=$1
VERSION=19

set -e
. ${PROJECT_SRC_DIR}/utils/scripts/functions.sh

echo "### Remove old ISL, get a new one and decompress it. ####"

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/3rdParty/"
echo_and_run_cmd "rm -rf isl/"
echo_and_run_cmd "wget http://isl.gforge.inria.fr/isl-0.$VERSION.tar.gz"
echo_and_run_cmd "tar -xvf isl-0.$VERSION.tar.gz"
echo_and_run_cmd "rm -rf isl-0.$VERSION.tar.gz"
echo_and_run_cmd "mv isl-0.$VERSION isl/"
