set(CMAKE_BUILD_TYPE DEBUG)

# Set to TRUE if you wish to use GPU
set(USE_GPU FALSE)

# Set to TRUE if you wish to use MPI
set(USE_MPI FALSE)

# Set to TRUE if you wish to use libpng tutorials and benchmarks
set(USE_LIBPNG FALSE)

# If USE_MPI is true, you need to set MPI paths
# (MPI_INCLUDE_DIR, MPI_LIB_DIR, and MPI_LIB_FLAGS)
# Examples:
# set(MPI_INCLUDE_DIR "/usr/local/Cellar/open-mpi/3.0.1/include/")
# set(MPI_LIB_DIR "/usr/local/Cellar/open-mpi/3.0.1/lib/")
# set(MPI_LIB_FLAGS "-lmpi")
# set(MPI_NODES "node1,node2")
set(MPI_INCLUDE_DIR "")
set(MPI_LIB_DIR "")
set(MPI_LIB_FLAGS "")
set(MPI_NODES "")

# Intel MKL library path. The specified folder should contain the folders
# include and lib.
# Example:
# set(MKL_PREFIX "/data/scratch/baghdadi/libs/intel/mkl/")
set(MKL_PREFIX "")

# LLVM_CONFIG_BIN: Directory containing llvm-config executable. Example:
# set(LLVM_CONFIG_BIN "/Users/je23693/Documents/external-code/llvm5.0/build/bin")
set(LLVM_CONFIG_BIN "${CMAKE_SOURCE_DIR}/3rdParty/llvm/prefix/bin/")

# ISL paths
set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")

# Halide Paths
set(HALIDE_SOURCE_DIRECTORY "3rdParty/Halide")
set(HALIDE_LIB_DIRECTORY "3rdParty/Halide/lib")

