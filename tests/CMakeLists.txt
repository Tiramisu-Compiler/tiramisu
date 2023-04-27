# Find package?
function(add_tiramisu_gen_test NUM)
add_tiramisu_generator("test${NUM}gen" SOURCES "test_${NUM}.cpp")
add_tiramisu_test_library("test${NUM}lib" FROM "test${NUM}gen" FUNCTION_NAME "generated_fct_test_${NUM}")
set_target_properties("test${NUM}lib" PROPERTIES EXCLUDE_FROM_ALL true)



add_executable("test${NUM}" "wrapper_test_${NUM}.cpp" "wrapper_test_${NUM}.h")
target_link_libraries("test${NUM}" "test${NUM}lib")
add_test(NAME "test${NUM}_build" COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target "test${NUM}")
add_test(NAME "test_${NUM}" COMMAND "test${NUM}" PROPERTIES DEPENDS "test${NUM}_build")

endfunction()

add_subdirectory(correctness)
