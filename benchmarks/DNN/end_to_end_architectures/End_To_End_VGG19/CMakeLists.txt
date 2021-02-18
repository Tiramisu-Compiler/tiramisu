cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -fopenmp -std=c++11 -O3 -fno-rtti -lmkldnn")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(compare_SOURCES end_to_end_vgg19_tiramisu_wrapper.cpp)

add_executable(wrapper_end_to_end_vgg19_tiramisu ${compare_SOURCES})

target_link_libraries(wrapper_end_to_end_vgg19_tiramisu ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_sparse_conv_relu_conv_relu_256_512_28_tiramisu.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_sparse_conv_relu_conv_relu_512_512_14_tiramisu.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_sparse_vgg_block_512_512_14_tiramisu.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_sparse_vgg_block_512_512_28_tiramisu.o -I/usr/local/include -L/usr/local/lib -lmkldnn tiramisu)
