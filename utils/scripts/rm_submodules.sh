#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: rm_submodules.sh <TIRAMISU_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=$1

set -e
. ${PROJECT_SRC_DIR}/utils/scripts/functions.sh
set +e

echo_and_run_cmd "cd ${PROJECT_SRC_DIR}"
echo_and_run_cmd "rm -rf 3rdParty/isl"
echo_and_run_cmd "rm -rf 3rdParty/Halide"
echo_and_run_cmd "rm -rf 3rdParty/llvm"
