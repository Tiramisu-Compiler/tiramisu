cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -fopenmp -std=c++11 -O3 -fno-rtti -lmkldnn")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(compare_SOURCES end_to_end_resnet_tiramisu_wrapper.cpp)

add_executable(wrapper_end_to_end_resnet_tiramisu ${compare_SOURCES})

target_link_libraries(wrapper_end_to_end_resnet_tiramisu ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_spconv_relu_maxpool.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block16_16.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block32_32.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block64_64.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block_16_16_stride2.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block_16_32_stride2.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/generated_fused_sparse_resnet_block_32_64_stride2.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/add_relu/add_relu_inplace_32_16_58_tiramisu.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/add_relu/add_relu_inplace_32_32_30_tiramisu.o ${CMAKE_CURRENT_SOURCE_DIR}/tiramisu_functions/add_relu/add_relu_inplace_32_64_16_tiramisu.o -I/usr/local/include -L/usr/local/lib -lmkldnn tiramisu)
