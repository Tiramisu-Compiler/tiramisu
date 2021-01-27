cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES sparse_vgg_block_32_generator.cpp)
set(compare_SOURCES sparse_vgg_block_32_wrapper.cpp)

add_executable(sparse_vgg_block_32_generator ${tiramisu_SOURCES})
add_executable(sparse_vgg_block_32_wrapper ${compare_SOURCES})

target_link_libraries(sparse_vgg_block_32_generator tiramisu)
target_link_libraries(sparse_vgg_block_32_wrapper ${CMAKE_CURRENT_SOURCE_DIR}/generated_sparse_vgg_block_32.o tiramisu)

add_custom_target(run_sparse_vgg_block_32
  COMMAND sparse_vgg_block_32_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(sparse_vgg_block_32_wrapper run_sparse_vgg_block_32)
