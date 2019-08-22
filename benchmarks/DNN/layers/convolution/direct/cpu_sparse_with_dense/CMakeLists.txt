cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#-Wl,--no-as-needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti -I.")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include /Users/b/Documents/src/MIT/tiramisu/benchmarks/DNN/layers/convolution/direct/cpu_sparse_with_dense)

set(tiramisu_SOURCES conv_layer_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_conv_layer.cpp)

add_executable(conv_sparse_with_dense_layer_generator ${tiramisu_SOURCES})
add_executable(wrapper_conv_sparse_with_dense_layer ${compare_SOURCES})
 
target_link_libraries(conv_sparse_with_dense_layer_generator tiramisu)
target_link_libraries(wrapper_conv_sparse_with_dense_layer ${CMAKE_CURRENT_SOURCE_DIR}/generated_conv_layer.o tiramisu)

add_custom_target(run_conv_sparse_with_dense_layer_generator
  COMMAND conv_sparse_with_dense_layer_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_conv_sparse_with_dense_layer run_conv_sparse_with_dense_layer_generator)
