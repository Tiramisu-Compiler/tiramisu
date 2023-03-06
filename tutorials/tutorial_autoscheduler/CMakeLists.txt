cmake_minimum_required(VERSION 3.5)
include(CMakePrintHelpers)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_VERBOSE_MAKEFILE "ON")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0")


include_directories($ENV{TIRAMISU_ROOT}/3rdParty/Halide/include $ENV{TIRAMISU_ROOT}/include/ $ENV{TIRAMISU_ROOT}/3rdParty/isl/include)

link_directories($ENV{TIRAMISU_ROOT}/build $ENV{TIRAMISU_ROOT}/3rdParty/Halide/lib $ENV{TIRAMISU_ROOT}/3rdParty/isl/build/lib)
cmake_print_variables(BUILD_TARGET)
if ($ENV{BUILD_TARGET} STREQUAL "autoscheduler")
    set(generator_SOURCES $ENV{FUNCTION_TO_BUILD}_autoscheduler.cpp)
elseif($ENV{BUILD_TARGET} STREQUAL "generator")
    set(generator_SOURCES $ENV{FUNCTION_TO_BUILD}_generator.cpp)
endif()

add_executable(generator ${generator_SOURCES})
 
#target_link_libraries(generator tiramisu tiramisu_auto_scheduler)
target_link_libraries(generator tiramisu tiramisu_auto_scheduler Halide isl)
