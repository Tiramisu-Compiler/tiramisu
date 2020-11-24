cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES conv_relu_conv_relu_generator.cpp)
set(compare_SOURCES conv_relu_conv_relu_wrapper.cpp)

add_executable(sparse_conv_relu_conv_relu_generator ${tiramisu_SOURCES})
add_executable(wrapper_sparse_conv_relu_conv_relu ${compare_SOURCES})

target_link_libraries(sparse_conv_relu_conv_relu_generator tiramisu)
target_link_libraries(wrapper_sparse_conv_relu_conv_relu ${CMAKE_CURRENT_SOURCE_DIR}/generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o tiramisu)

add_custom_target(run_sparse_conv_relu_conv_relu
  COMMAND sparse_conv_relu_conv_relu_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_sparse_conv_relu_conv_relu run_sparse_conv_relu_conv_relu)
