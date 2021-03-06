cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES conv_relu_maxpool_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_nn_block.cpp)

add_executable(conv_relu_maxpool_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_nn_block_conv_relu_maxpool ${compare_SOURCES})
 
target_link_libraries(conv_relu_maxpool_generator_tiramisu tiramisu)
target_link_libraries(wrapper_nn_block_conv_relu_maxpool ${CMAKE_CURRENT_SOURCE_DIR}/conv_relu_maxpool_tiramisu.o tiramisu)

add_custom_target(run_conv_relu_maxpool
  COMMAND conv_relu_maxpool_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_nn_block_conv_relu_maxpool run_conv_relu_maxpool)
