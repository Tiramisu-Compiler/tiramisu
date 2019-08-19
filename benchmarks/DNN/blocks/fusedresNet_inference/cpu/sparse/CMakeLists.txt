cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES fused_sparse_resnet_block_generator.cpp)
set(compare_SOURCES fused_sparse_resnet_block_wrapper.cpp)

add_executable(fused_sparse_resnet_block_generator ${tiramisu_SOURCES})
add_executable(fused_sparse_resnet_block_wrapper ${compare_SOURCES})

target_link_libraries(fused_sparse_resnet_block_generator tiramisu)
target_link_libraries(fused_sparse_resnet_block_wrapper ${CMAKE_CURRENT_SOURCE_DIR}/generated_fused_sparse_resnet_block.o tiramisu)

add_custom_target(run_fused_sparse_resnet_block
  COMMAND fused_sparse_resnet_block_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(fused_sparse_resnet_block_wrapper run_fused_sparse_resnet_block)
