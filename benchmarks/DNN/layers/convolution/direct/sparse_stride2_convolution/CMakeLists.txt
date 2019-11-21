cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES stride2_spconv_generator.cpp)
set(compare_SOURCES stride2_spconv_wrapper.cpp)

add_executable(stride2_spconv_generator ${tiramisu_SOURCES})
add_executable(stride2_spconv_wrapper ${compare_SOURCES})

target_link_libraries(stride2_spconv_generator tiramisu)
target_link_libraries(stride2_spconv_wrapper ${CMAKE_CURRENT_SOURCE_DIR}/generated_stride2_spconv.o tiramisu)

add_custom_target(run_stride2_spconv
  COMMAND stride2_spconv_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(stride2_spconv_wrapper run_stride2_spconv)
