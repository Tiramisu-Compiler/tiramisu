set(benchmark_name benchmark_lstm_gpu_lib)
set(generator_name ${benchmark_name}_generator)
set(wrapper_name ${benchmark_name}_wrapper)
set(object_files lstm.o lstm.o_gpu.o lstm.o_cpu.o)

add_executable(${generator_name} generator.cpp)
target_link_libraries(${generator_name} tiramisu ${HalideLib} ${ISLLib} ${LINK_FLAGS})
add_custom_command(OUTPUT ${object_files} COMMAND ${generator_name} DEPENDS ${generator_name})

add_executable(${wrapper_name} wrapper.cpp ${object_files})
target_link_libraries(${wrapper_name} tiramisu ${HalideLib} ${ISLLib} ${LINK_FLAGS} cuda_wrapper ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES})
target_include_directories(${wrapper_name} PUBLIC ${CUDNN_INCLUDE_DIRECTORY})

add_custom_target(run_${benchmark_name} COMMAND ${wrapper_name})
add_custom_target(run_${benchmark_name}_correctness COMMAND ${wrapper_name} 1 1 0 1)
add_custom_target(run_${benchmark_name}_nvprof COMMAND LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvprof --profile-from-start off --print-gpu-trace $<TARGET_FILE:${wrapper_name}> DEPENDS ${wrapper_name})
add_custom_target(run_${benchmark_name}_nvprof2 COMMAND LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvprof --profile-from-start off $<TARGET_FILE:${wrapper_name}> DEPENDS ${wrapper_name})
