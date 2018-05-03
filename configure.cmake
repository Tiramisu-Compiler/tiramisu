# Comment out this message if you are done configuring this file
set(NEEDS_CONFIG true)

if (${NEEDS_CONFIG})
    message(WARNING "Please make sure to configure configure.cmake, and then comment out the second line.")
endif()

set(LLVM_CONFIG_BIN "")

set(ISL_INCLUDE_DIRECTORY "3rdParty/isl/build/include/")
set(ISL_LIB_DIRECTORY "3rdParty/isl/build/lib/")
set(HALIDE_SOURCE_DIRECTORY "Halide")
set(HALIDE_LIB_DIRECTORY "Halide/lib")

set(MKL_PREFIX "")

# EXAMPLE OF SETTING MPI PATHS
#set(MPI_INCLUDE_DIR "/usr/local/Cellar/open-mpi/3.0.1/include/")
#set(MPI_LIB_DIR "/usr/local/Cellar/open-mpi/3.0.1/lib/")
#set(MPI_LIB_FLAGS "-lmpi")
set(MPI_INCLUDE_DIR "")
set(MPI_LIB_DIR "")
set(MPI_LIB_FLAGS "")

# Uncomment if you wish to use GPU
# set(USE_GPU true)

# Uncomment if you wish to use MPI
# set(USE_MPI true)

