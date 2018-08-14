cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3  -fno-rtti")

include_directories( ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/3rdParty/isl/include )

set(halide_SOURCES conv_layer_generator.cpp)
set(tiramisu_SOURCES conv_layer_generator_tiramisu.cpp)
set(compare_SOURCES wrapper_nn.cpp)

add_executable(conv_layer_generator ${halide_SOURCES})
add_executable(conv_layer_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_nn ${compare_SOURCES})
 
target_link_libraries(conv_layer_generator  ${PROJECT_DIR}/3rdParty/Halide/lib/libHalide.a )
target_link_libraries(conv_layer_generator_tiramisu   ${PROJECT_DIR}/build/libtiramisu.so)
target_link_libraries(wrapper_nn ${CMAKE_CURRENT_SOURCE_DIR}/generated_conv.o ${CMAKE_CURRENT_SOURCE_DIR}/generated_conv_tiramisu.o ${PROJECT_DIR}/build/libtiramisu.so 
)

add_custom_target(halide_run
  COMMAND conv_layer_generator 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated halide executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/conv_layer_generator.cpp 
)

add_custom_target(tiramisu_run
  COMMAND conv_layer_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/conv_layer_generator_tiramisu.cpp 
)

add_dependencies(wrapper_nn halide_run)
add_dependencies(wrapper_nn tiramisu_run)
