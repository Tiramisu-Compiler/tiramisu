RED='\033[0;31m'
NC='\033[0m' # No Color
# TODO after configuring this file, please comment out the next line.
echo -e "${RED}Please make sure to configure configure_paths.sh${NC}"


# Path to the LLVM prefix folder that contain llvm-config.
LLVM_CONFIG_BIN=""

# Example
#LLVM_CONFIG_BIN=/Users/b/Documents/src-not-saved/llvm/llvm_39_prefix/bin/

############################################################
# No need to change the following variable paths unless you
# installed Tiramisu without using the installation scripts.
ISL_INCLUDE_DIRECTORY=3rdParty/isl/build/include/
ISL_LIB_DIRECTORY=3rdParty/isl/build/lib/
HALIDE_SOURCE_DIRECTORY=Halide
HALIDE_LIB_DIRECTORY=Halide/lib

############################################################
# Set the path to MKL if you want to run benchmarks.
MKL_PREFIX=/opt/intel/compilers_and_libraries/mac/mkl/
