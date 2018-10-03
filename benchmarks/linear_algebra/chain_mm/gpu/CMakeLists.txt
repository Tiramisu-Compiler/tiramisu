set(benchmark_name benchmark_chain_mm_gpu)
set(generator_name ${benchmark_name}_generator)
set(wrapper_name ${benchmark_name}_wrapper)
set(object_files fct.o fct.o_cpu.o fct.o_gpu.o)

add_executable(${generator_name} generator.cpp)
target_link_libraries(${generator_name} tiramisu ${HalideLib} ${ISLLib} ${LINK_FLAGS} cuda_wrapper)
add_custom_command(OUTPUT ${object_files} COMMAND ${generator_name} DEPENDS ${generator_name})

add_executable(${wrapper_name} wrapper.cpp ${object_files})
# Override optimization flag to make correctness fast:
set_target_properties(${wrapper_name} PROPERTIES COMPILE_FLAGS -O3)
target_link_libraries(${wrapper_name} tiramisu ${HalideLib} ${ISLLib} ${LINK_FLAGS} cuda_wrapper ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

set(testN 100)

add_custom_target(run_${benchmark_name} COMMAND ${wrapper_name} ${testN} 0 DEPENDS ${wrapper_name})
add_custom_target(run_${benchmark_name}_nvprof COMMAND LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvprof --print-gpu-trace $<TARGET_FILE:${wrapper_name}> 1 0 DEPENDS ${wrapper_name})
add_custom_target(run_${benchmark_name}_correctness COMMAND ${wrapper_name} 0 1 DEPENDS ${wrapper_name})
