cmake_minimum_required(VERSION 3.5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -ldl -g -lz -lpthread -std=c++11 -O3 -fno-rtti")

include_directories(${PROJECT_DIR}/3rdParty/Halide/include ${PROJECT_DIR}/include/ ${PROJECT_DIR}/3rdParty/isl/include)

set(tiramisu_SOURCES generator.cpp)
set(compare_SOURCES wrapper.cpp)

add_executable(lstm_generator_tiramisu ${tiramisu_SOURCES})
add_executable(wrapper_lstm ${compare_SOURCES})
 
target_link_libraries(lstm_generator_tiramisu tiramisu)
target_link_libraries(wrapper_lstm ${CMAKE_CURRENT_SOURCE_DIR}/lstm.o tiramisu mkl_wrapper -DMKL_ILP64 -m64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm)

add_custom_target(run_lstm
  COMMAND lstm_generator_tiramisu
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "run generated tiramisu executable in ${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES ${tiramisu_SOURCES}
)
add_dependencies(wrapper_lstm run_lstm)
