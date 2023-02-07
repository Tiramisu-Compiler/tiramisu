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

add_library(tiramisu SHARED ${SOURCES} )
target_link_libraries(tiramisu ${HalideLib} ${ISLLib})
target_link_libraries(tiramisu ${LINK_FLAGS})
