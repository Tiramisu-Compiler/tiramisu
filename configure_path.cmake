# Path to the LLVM prefix folder that contain llvm-config.
set(LLVM_CONFIG_BIN "")

# Example
#set(LLVM_CONFIG_BIN "/Users/b/Documents/src-not-saved/llvm/llvm-3.7.0_prefix/bin/")

############################################################
# No need to change the following variable paths unless you
# installed Tiramisu without using the installation scripts.
set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")
set(HALIDE_SOURCE_DIRECTORY "Halide")
set(HALIDE_LIB_DIRECTORY "Halide/lib")

############################################################
# Set the path to MKL if you want to run benchmarks.
set(MKL_PREFIX "/opt/intel/compilers_and_libraries/mac/mkl/")
