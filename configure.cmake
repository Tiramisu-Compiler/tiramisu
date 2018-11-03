set(CMAKE_BUILD_TYPE DEBUG)

# Set to TRUE if you wish to use GPU
set(USE_GPU FALSE)

# Set to TRUE if you wish to use MPI
set(USE_MPI FALSE)

# Set to TRUE if you wish to use libpng which is needed only by the Halide benchmarks
set(USE_LIBPNG FALSE)

# Set to TRUE if you wish to use libjpeg which is needed only by the Halide benchmarks
set(USE_LIBJPEG FALSE)

# If USE_MPI is true, you need to the MPI_BUILD_DIR and MPI_NODES path
# Note: This assumes you are using your own installed version of MPI. If your system already
# has a version of openmpi installed, you will have to read the docs to see what the appropriate
# way of launching mpi jobs is. For our testing, we use mpirun.
# Examples:
#set(MPI_BUILD_DIR "/data/scratch/jray/Repositories/tiramisu/3rdParty/openmpi-3.1.2/build/")
#set(MPI_NODES "lanka01,lanka02,lanka03,lanka04,lanka05,lanka06,lanka12,lanka13,lanka14,lanka15")
set(MPI_BUILD_DIR "")
set(MPI_NODES "")

# Intel MKL library path. The specified folder should contain the folders
# include and lib.
# Example:
# set(MKL_PREFIX "/data/scratch/baghdadi/libs/intel/mkl/")
set(MKL_PREFIX "")

# LLVM_CONFIG_BIN: Directory containing llvm-config executable.
set(LLVM_CONFIG_BIN "${CMAKE_SOURCE_DIR}/3rdParty/llvm/prefix/bin/")

# ISL paths
set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")

# Halide Paths
set(HALIDE_SOURCE_DIRECTORY "3rdParty/Halide")
set(HALIDE_LIB_DIRECTORY "3rdParty/Halide/lib")

