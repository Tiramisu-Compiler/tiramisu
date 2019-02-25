cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES relu_layer_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_nn_relu.cpp)

add_executable(relu_layer_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_nn_relu ${compare_SOURCES})
 
target_link_libraries(relu_layer_generator_tiramisu tiramisu)
target_link_libraries(wrapper_nn_relu ${CMAKE_CURRENT_SOURCE_DIR}/relu_layer_generator_tiramisu.o tiramisu)

add_custom_target(run_relu
  COMMAND relu_layer_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES} 
)

add_dependencies(wrapper_nn_relu run_relu)
