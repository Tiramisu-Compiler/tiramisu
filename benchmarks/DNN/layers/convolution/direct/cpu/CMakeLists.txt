cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -g -lz -lpthread -std=c++11 -O3  -fno-rtti")

include_directories( ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/3rdParty/isl/include )

set(tiramisu_SOURCES conv_layer_generator.cpp)
set(compare_SOURCES conv_layer_wrapper.cpp)

add_executable(conv_layer_generator ${tiramisu_SOURCES})
add_executable(conv_layer_wrapper ${compare_SOURCES})
 
target_link_libraries(conv_layer_generator tiramisu)
target_link_libraries(conv_layer_wrapper ${CMAKE_CURRENT_SOURCE_DIR}/generated_conv_layer.o tiramisu)

add_custom_target(run_conv_layer
  COMMAND conv_layer_wrapper
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/conv_layer_generator.cpp)
