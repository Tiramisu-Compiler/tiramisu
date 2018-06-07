set(CMAKE_BUILD_TYPE RELEASE)
# directory containing llvm-config executable
# example of setting LLVM_CONFIG_BIN
# set(LLVM_CONFIG_BIN "/Users/je23693/Documents/external-code/llvm5.0/build/bin")
set(LLVM_CONFIG_BIN "")

set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")
set(HALIDE_SOURCE_DIRECTORY "Halide")
set(HALIDE_LIB_DIRECTORY "Halide/lib")

set(MKL_PREFIX "")

# Uncomment if you wish to use GPU
# set(USE_GPU true)

# Uncomment if you wish to use MPI
# set(USE_MPI true)

# If USE_MPI==true, you need to set these paths
# example of setting MPI paths
#set(MPI_INCLUDE_DIR "/usr/local/Cellar/open-mpi/3.0.1/include/")
#set(MPI_LIB_DIR "/usr/local/Cellar/open-mpi/3.0.1/lib/")
#set(MPI_LIB_FLAGS "-lmpi")
set(MPI_INCLUDE_DIR "")
set(MPI_LIB_DIR "")
set(MPI_LIB_FLAGS "")
