# Find package?
function(add_tiramisu_gen_test NUM)
add_tiramisu_generator("test${NUM}gen" SOURCES "test_${NUM}.cpp")
add_tiramisu_library("test${NUM}lib" FROM "test${NUM}gen" FUNCTION_NAME "generated_fct_test_${NUM}")

add_executable("test${NUM}" "wrap_test_${NUM}.cpp" "wrapper_test_${NUM}.h")
target_link_libraries("test${NUM}" "test${NUM}lib")

add_test(NAME "test_${NUM}" COMMAND "test${NUM}")

endfunction()

add_subdirectory(correctness)
