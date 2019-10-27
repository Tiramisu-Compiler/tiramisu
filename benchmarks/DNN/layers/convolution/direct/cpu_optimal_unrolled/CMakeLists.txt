cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES conv_layer_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_conv_layer.cpp)

add_executable(cpu_optimal_unrolled_generator ${tiramisu_SOURCES})
add_executable(wrapper_cpu_optimal_unrolled ${compare_SOURCES})
 
target_link_libraries(cpu_optimal_unrolled_generator tiramisu)
target_link_libraries(wrapper_cpu_optimal_unrolled ${CMAKE_CURRENT_SOURCE_DIR}/generated_conv_layer.o tiramisu)

add_custom_target(run_cpu_optimal_unrolled_generator
  COMMAND cpu_optimal_unrolled_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_cpu_optimal_unrolled run_cpu_optimal_unrolled_generator)
