set(CMAKE_BUILD_TYPE DEBUG)

option(CMAKE_C_COMPILER "clang" ON)
option(CMAKE_CXX_COMPILER "clang++" ON)

option(USE_GPU "Build with GPU support" OFF)

option(USE_MPI "Build with MPI support" OFF)
option(USE_HALIDE "Build with Halide support" OFF)

option(USE_LIBPNG "Build with libpng for the Halide benchmark" FALSE)

option(USE_LIBJPEG "Build with libjpeg for the Halide benchmark" FALSE)

option(USE_CUDNN "Build with cuDNN for benchmark comparisons" FALSE)

# If you set this to true, you should correctly set MKL_PREFIX (see below)
option(USE_MKL_WRAPPERS "Build with MKL wrappers provided by Tiramisu" FALSE)

# Change with the cudnn library location
set(CUDNN_LOCATION /data/scratch/akkas/cudnn7 CACHE PATH "CUDNN library location")

# If USE_MPI is true, you need to the MPI_BUILD_DIR and MPI_NODES path
# Note: This assumes you are using your own installed version of MPI. If your system already
# has a version of openmpi installed, you will have to read the docs to see what the appropriate
# way of launching mpi jobs is. For our testing, we use mpirun.
# Examples:
#set(MPI_BUILD_DIR "/data/scratch/jray/Repositories/tiramisu/3rdParty/openmpi-3.1.2/build/")
#set(MPI_NODES "lanka01,lanka02,lanka03,lanka04,lanka05,lanka06,lanka12,lanka13,lanka14,lanka15")
set(MPI_BUILD_DIR "" CACHE PATH "Build directory of MPI")
set(MPI_NODES "" CACHE PATH "Use of MPI node paths")

# The specified folder should contain the folders include and lib.
# Example:
# set(MKL_PREFIX "/data/scratch/baghdadi/libs/intel/mkl/")
set(MKL_PREFIX "" CACHE PATH "Intel MKL library path")

#set(LLVM_CONFIG_BIN "${CMAKE_SOURCE_DIR}/3rdParty/llvm/prefix/bin/" CACHE PATH "Directory containing llvm-config executable")
set(LLVM_CONFIG_BIN "/usr/bin/" CACHE PATH "Directory containing llvm-config executable")

# Debug
option(ENABLE_DEBUG "Enable debug printing" FALSE)
set(DEBUG_LEVEL 0 CACHE STRING "Debug level value")

# ISL paths
set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/" CACHE PATH "Path to ISL include directory")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/" CACHE PATH "Path to ISL library directory")

# Halide Paths
set(HALIDE_SOURCE_DIRECTORY "3rdParty/Halide/build" CACHE PATH "Path to Halide source directory")
set(HALIDE_INCLUDE_DIRECTORY "3rdParty/Halide/build/include" CACHE PATH "Path to Halide include directory")
set(HALIDE_LIB_DIRECTORY "3rdParty/Halide/build/lib" CACHE PATH "Path to Halide library directory")
