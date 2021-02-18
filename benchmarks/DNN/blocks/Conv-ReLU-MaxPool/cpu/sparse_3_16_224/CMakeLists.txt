cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES spconv_relu_maxpool_generator.cpp)
set(compare_SOURCES spconv_relu_maxpool_wrapper.cpp)

add_executable(spconv_relu_maxpool_generator_3_16_224 ${tiramisu_SOURCES})
add_executable(spconv_relu_maxpool_wrapper_3_16_224 ${compare_SOURCES})

target_link_libraries(spconv_relu_maxpool_generator_3_16_224 tiramisu)
target_link_libraries(spconv_relu_maxpool_wrapper_3_16_224 ${CMAKE_CURRENT_SOURCE_DIR}/generated_spconv_relu_maxpool.o tiramisu)

add_custom_target(run_spconv_relu_maxpool_3_16_224
  COMMAND spconv_relu_maxpool_generator_3_16_224
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(spconv_relu_maxpool_wrapper_3_16_224 run_spconv_relu_maxpool_3_16_224)
