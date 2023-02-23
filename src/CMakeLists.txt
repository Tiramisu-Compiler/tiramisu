set(SOURCES
tiramisu_expr.cpp
tiramisu_block.cpp
tiramisu_core.cpp
tiramisu_codegen_halide.cpp
tiramisu_codegen_c.cpp
tiramisu_debug.cpp
tiramisu_function.cpp
tiramisu_utils.cpp
tiramisu_codegen_halide_lowering.cpp
tiramisu_codegen_from_halide.cpp
tiramisu_mpi.cpp
tiramisu_codegen_cuda.cpp
tiramisu_externs.cpp
)

set(HEADERS
${CMAKE_SOURCE_DIR}/include/tiramisu/block.h
${CMAKE_SOURCE_DIR}/include/tiramisu/core.h
${CMAKE_SOURCE_DIR}/include/tiramisu/cuda_ast.h
${CMAKE_SOURCE_DIR}/include/tiramisu/debug.h
${CMAKE_SOURCE_DIR}/include/tiramisu/expr.h
${CMAKE_SOURCE_DIR}/include/tiramisu/externs.h
${CMAKE_SOURCE_DIR}/include/tiramisu/macros.h
${CMAKE_SOURCE_DIR}/include/tiramisu/mpi_comm.h
${CMAKE_SOURCE_DIR}/include/tiramisu/type.h
${CMAKE_SOURCE_DIR}/include/tiramisu/utils.h
${CMAKE_SOURCE_DIR}/include/tiramisu/tiramisu.h
)

add_library(tiramisu SHARED ${SOURCES})
target_link_libraries(tiramisu Halide::Halide ${ISLLib})
target_link_libraries(tiramisu ${LINK_FLAGS})
set_target_properties(tiramisu
  PROPERTIES
  LIBRARY_OUTPUT_NAME tiramisu
  EXPORT_NAME tiramisu
  PUBLIC_HEADER "${HEADERS}"
  )
target_include_directories(tiramisu PUBLIC ${CMAKE_SOURCE_DIR}/include/tiramisu)

if (${USE_AUTO_SCHEDULER})
add_subdirectory(auto_scheduler)
endif()
