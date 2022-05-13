cmake_minimum_required(VERSION 3.5)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_VERBOSE_MAKEFILE "ON")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0")

include_directories($ENV{TIRAMISU_ROOT}/3rdParty/Halide/include $ENV{TIRAMISU_ROOT}/include/ $ENV{TIRAMISU_ROOT}/3rdParty/isl/include)

link_directories($ENV{TIRAMISU_ROOT}/build $ENV{TIRAMISU_ROOT}/3rdParty/Halide/lib $ENV{TIRAMISU_ROOT}/3rdParty/isl/build/lib)


set(generator_SOURCES function_heat3d_MINI_autoscheduler.cpp)
#set(generator_SOURCES function_heat3d_MINI_generator.cpp)
add_executable(generator ${generator_SOURCES})
 
target_link_libraries(generator tiramisu tiramisu_auto_scheduler Halide isl)
