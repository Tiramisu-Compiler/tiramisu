cmake_minimum_required(VERSION 3.7)
project(restructured)

set(CMAKE_CXX_STANDARD 11)

add_executable(restructured main.cpp classes.cpp classes.h tiramisu_code_generator.cpp tiramisu_code_generator.h external_writers.cpp external_writers.h)