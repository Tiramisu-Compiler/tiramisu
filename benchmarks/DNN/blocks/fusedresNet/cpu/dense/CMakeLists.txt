cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES fused_resnet_block_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_nn_block.cpp)

add_executable(fused_resnet_block_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_nn_block_fused_resnet ${compare_SOURCES})
 
target_link_libraries(fused_resnet_block_generator_tiramisu tiramisu)
target_link_libraries(wrapper_nn_block_fused_resnet ${CMAKE_CURRENT_SOURCE_DIR}/fused_resnet_block_generator_tiramisu.o tiramisu)

add_custom_target(run_fused_resnet_block
  COMMAND fused_resnet_block_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_nn_block_fused_resnet run_fused_resnet_block)
