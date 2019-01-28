cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(  ${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include )

set(tiramisu_SOURCES vgg_block_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_nn_block.cpp)

add_executable(vgg_block_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_nn_block_vgg ${compare_SOURCES})
 
target_link_libraries(vgg_block_generator_tiramisu tiramisu)
target_link_libraries(wrapper_nn_block_vgg ${CMAKE_CURRENT_SOURCE_DIR}/generated_vgg_block.o tiramisu)

add_custom_target(tiramisu_run_vgg_block
  COMMAND vgg_block_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES  ${tiramisu_SOURCES} 
)

add_dependencies(wrapper_nn_block_vgg tiramisu_run_vgg_block)
