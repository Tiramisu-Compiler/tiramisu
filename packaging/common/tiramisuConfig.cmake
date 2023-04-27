cmake_minimum_required(VERSION 3.22)

find_package(Halide REQUIRED)
find_library(ISLLib isl PATHS ${ISL_LIB_DIRECTORY} NO_DEFAULT_PATH)

set(tiramisu_FOUND TRUE)


if (${USE_GPU})
    find_package(CUDA REQUIRED)
    if (${USE_CUDNN})
        find_library(CUDNN_LIBRARIES cudnn PATHS ${CUDNN_LOCATION}/lib64 NO_DEFAULT_PATH)
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/TiramisuGeneratorHelpers.cmake)

@PACKAGE_INIT@

include(${CMAKE_CURRENT_LIST_DIR}/TiramisuTargets.cmake)
