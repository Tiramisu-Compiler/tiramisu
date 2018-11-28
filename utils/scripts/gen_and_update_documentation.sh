#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage: gen_and_update_documentation.sh <TIRAMISU_ROOT_PATH> <TIRAMISU_WEBSITE_ROOT_PATH>"
	exit 1
fi

PROJECT_SRC_DIR=$1
WEBSITE_DIR=$2

set -e
. ${PROJECT_SRC_DIR}/utils/scripts/functions.sh

echo "### Generate new documentation, copy the generated documentation to the website. ####"

echo_and_run_cmd "cd ${WEBSITE_DIR}"
echo_and_run_cmd "git pull"
echo_and_run_cmd "cd ${PROJECT_SRC_DIR}/build/"
echo_and_run_cmd "make doc"
echo_and_run_cmd "cp -r doc/* ${WEBSITE_DIR}/doc/"
echo_and_run_cmd "cd ${WEBSITE_DIR}"
echo_and_run_cmd "git add doc"
echo_and_run_cmd "git commit -m \"update\""
echo_and_run_cmd "git pull"
echo_and_run_cmd "git push"
