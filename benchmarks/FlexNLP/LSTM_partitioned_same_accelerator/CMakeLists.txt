cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES flexnlp_tiramisu_generator.cpp)
set(compare_SOURCES flexnlp_tiramisu_wrapper.cpp)

add_executable(flexnlp_tiramisu_lstm_partitioned_generator ${tiramisu_SOURCES})
add_executable(wrapper_flexnlp_tiramisu_lstm_partitioned ${compare_SOURCES})

target_link_libraries(flexnlp_tiramisu_lstm_partitioned_generator tiramisu)
target_link_libraries(wrapper_flexnlp_tiramisu_lstm_partitioned ${CMAKE_CURRENT_SOURCE_DIR}/generated_flexnlp_test.o tiramisu ${HalideLib} ${ISLLib} ${LINK_FLAGS} flexnlp_wrapper)

add_custom_target(run_flexnlp_tiramisu_partitioned_lstm
  COMMAND flexnlp_tiramisu_lstm_partitioned_generator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_flexnlp_tiramisu_lstm_partitioned run_flexnlp_tiramisu_partitioned_lstm)
