set(CMAKE_BUILD_TYPE DEBUG)

option(USE_GPU "Build with GPU support" OFF)

option(USE_MPI "Build with MPI support" OFF)

option(USE_CUDNN "Build with cuDNN for benchmark comparisons" FALSE)

option(WITH_PYTHON_BINDINGS "Build Python bindings" ON)

option(WITH_TUTORIALS "Build Tutorials" OFF)

option(WITH_BENCHMAKRS "Build Benchmarks" OFF)

option(WITH_TESTS "Build Tests" OFF)

option(WITH_DOCS "Build Docs" OFF)



# If you set this to true, you should correctly set MKL_PREFIX (see below)
option(USE_MKL_WRAPPERS "Build with MKL wrappers provided by Tiramisu" FALSE)

option(USE_AUTO_SCHEDULER "Build the Tiramisu auto-scheduler" FALSE)

# Change with the cudnn library location
set(CUDNN_LOCATION /data/scratch/akkas/cudnn7 CACHE PATH "CUDNN library location")

# If USE_MPI is true, you need to the MPI_BUILD_DIR and MPI_NODES path
# Note: This assumes you are using your own installed version of MPI. If your system already
# has a version of openmpi installed, you will have to read the docs to see what the appropriate
# way of launching mpi jobs is. For our testing, we use mpirun.
# Examples:
set(MPI_BUILD_DIR "" CACHE PATH "Build directory of MPI")
set(MPI_NODES "" CACHE PATH "Use of MPI node paths")

# The specified folder should contain the folders include and lib.
# Example:
# set(MKL_PREFIX "/data/scratch/baghdadi/libs/intel/mkl/")
set(MKL_PREFIX "" CACHE PATH "Intel MKL library path")

# Debug
option(ENABLE_DEBUG "Enable debug printing" TRUE)
set(DEBUG_LEVEL 0 CACHE STRING "Debug level value")

# ISL paths
set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/" CACHE PATH "Path to ISL include directory")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/" CACHE PATH "Path to ISL library directory")