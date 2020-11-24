cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES fused_sparse_resnet_block_generator.cpp)
set(compare_SOURCES fused_sparse_resnet_block_wrapper.cpp)

add_executable(fused_sparse_resnet_block_generator16_16 ${tiramisu_SOURCES})
add_executable(fused_sparse_resnet_block_wrapper16_16 ${compare_SOURCES})

target_link_libraries(fused_sparse_resnet_block_generator16_16 tiramisu)
target_link_libraries(fused_sparse_resnet_block_wrapper16_16 ${CMAKE_CURRENT_SOURCE_DIR}/generated_fused_sparse_resnet_block16_16.o tiramisu)

add_custom_target(run_fused_sparse_resnet_block16_16
  COMMAND fused_sparse_resnet_block_generator16_16
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(fused_sparse_resnet_block_wrapper16_16 run_fused_sparse_resnet_block16_16)
