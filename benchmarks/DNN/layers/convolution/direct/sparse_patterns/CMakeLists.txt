cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES spconv_generator.cpp)
set(compare_SOURCES spconv_wrapper.cpp)

add_executable(spconv_generator_patterns ${tiramisu_SOURCES})
add_executable(spconv_wrapper_patterns ${compare_SOURCES})

target_link_libraries(spconv_generator_patterns tiramisu)
target_link_libraries(spconv_wrapper_patterns ${CMAKE_CURRENT_SOURCE_DIR}/generated_spconv.o tiramisu)

add_custom_target(run_spconv_patterns
  COMMAND spconv_generator_patterns
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(spconv_wrapper_patterns run_spconv_patterns)
