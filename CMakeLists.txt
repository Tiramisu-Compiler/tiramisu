cmake_minimum_required(VERSION 3.5)
project(tiramisu)

enable_testing()

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})

# Add CMake header files
set(HEADER_FILES
        include/tiramisu/core.h
        include/tiramisu/debug.h
        include/tiramisu/utils.h
        include/tiramisu/expr.h
        include/tiramisu/core.h
        )

# Add CMake cpp files
set(OBJS expr core codegen_halide codegen_c debug utils codegen_halide_lowering codegen_from_halide)


### CMAKE FILE INTERNALS ###

FILE(STRINGS test_list TIRAMISU_TESTS)
FILE(STRINGS benchmark_list TIRAMISU_BENCHMARKS)
FILE(STRINGS tutorial_list TIRAMISU_TUTORIALS)

include(configure_path.cmake)

set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11 -Wall -Wno-sign-compare -fno-rtti")# -fvisibility=hidden")
set(CMAKE_CXX_FLAGS_RELEASE "-g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti")# -fvisibility=hidden")

include_directories("include"
                    "${ISL_INCLUDE_DIRECTORY}"
                    "${HALIDE_SOURCE_DIRECTORY}/include" "${HALIDE_SOURCE_DIRECTORY}/tools"
                    "build")

find_library(HalideLib Halide PATHS ${HALIDE_LIB_DIRECTORY} NO_DEFAULT_PATH)
find_library(ISLLib isl PATHS ${ISL_LIB_DIRECTORY} NO_DEFAULT_PATH)

execute_process(COMMAND libpng-config --ldflags OUTPUT_VARIABLE LIBPNG_FLAGS)
execute_process(COMMAND ${LLVM_CONFIG_BIN}llvm-config --system-libs OUTPUT_VARIABLE LLVM_FLAGS)

string(STRIP ${LLVM_FLAGS} LLVM_FLAGS)
string(STRIP ${LIBPNG_FLAGS} LIBPNG_FLAGS)


set(LINK_FLAGS "-lgmp -ldl -lpthread -lz -ljpeg")
set(LINK_FLAGS "${LINK_FLAGS} ${LLVM_FLAGS} ${LIBPNG_FLAGS}")


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)

set(T_CODE "")

foreach (obj ${OBJS})
    set(T_CODE ${T_CODE} src/tiramisu_${obj}.cpp)
endforeach ()

add_library(tiramisu SHARED ${T_CODE})

function(build_g name generator result)
    add_executable(${name} ${generator} ${HEADER_FILES})
    target_link_libraries(${name} tiramisu ${HalideLib} ${ISLLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    if (NOT ${result})
        set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${result})
        add_custom_command(OUTPUT ${result} COMMAND ${name})
    endif()
endfunction()

function(build_halide_g name generator result)
    add_executable(${name} ${generator})
    target_link_libraries(${name} ${HalideLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    if (NOT ${result})
        set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${result})
        add_custom_command(OUTPUT ${result} COMMAND ${name})
    endif()
endfunction()

function(build_w name objs wrapper header)
    add_executable(${name} ${wrapper} ${objs} ${header})
    target_link_libraries(${name} tiramisu ${HalideLib} ${ISLLib})
    target_link_libraries(${name} ${LINK_FLAGS})
endfunction()


function(new_test id)
    set(generator_target test_${id}_fct_generator)
    set(generated_obj build/generated_fct_test_${id}.o)
    set(test_name test_${id})
    build_g(${generator_target} tests/test_${id}.cpp ${generated_obj})
    build_w(${test_name} "${generated_obj}" tests/wrapper_test_${id}.cpp tests/wrapper_test_${id}.h)
    add_test(NAME ${id}_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${test_name})
    add_test(NAME ${id} COMMAND ${test_name})
    set_tests_properties(${id} PROPERTIES DEPENDS ${id}_build)
endfunction()


build_g(test_global tests/test_global.cpp "")
add_test(NAME global_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target test_global)
add_test(NAME global COMMAND test_global)
set_tests_properties(global PROPERTIES DEPENDS global_build)

foreach(t ${TIRAMISU_TESTS})
 new_test(${t})
endforeach()

add_custom_target(benchmarks)

function(new_benchmark id)
    set(tiramisu_generator_target bench_tiramisu_${id}_generator)
    set(halide_generator_target   bench_halide_${id}_generator)
    set(generated_obj_tiramisu build/generated_fct_${id}.o)
    set(generated_obj_halide   build/generated_fct_${id}_ref.o)
    set(bench_name bench_${id})
    build_g(${tiramisu_generator_target} benchmarks/halide/${id}_tiramisu.cpp ${generated_obj_tiramisu})
    build_halide_g(${halide_generator_target} benchmarks/halide/${id}_ref.cpp ${generated_obj_halide})
    build_w(${bench_name} "${generated_obj_tiramisu};${generated_obj_halide}" benchmarks/halide/wrapper_${id}.cpp benchmarks/halide/wrapper_${id}.h)
    add_custom_target(run_benchmark_${id} COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    add_custom_command(TARGET benchmarks COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
endfunction()

foreach(b ${TIRAMISU_BENCHMARKS})
    new_benchmark(${b})
endforeach()

add_custom_target(doc DEPENDS doc/Doxyfile doc/Doxyfile_user doc/Doxyfile_dev)
add_custom_command(TARGET doc COMMAND doxygen doc/Doxyfile_user)
add_custom_command(TARGET doc COMMAND doxygen doc/Doxyfile_dev)

add_custom_target(tutorials)

function(new_tutorial id)
    set(generator_target tutorial_${id}_fct_generator)
    set(generated_obj build/generated_fct_tutorial_${id}.o)
    set(tutorial_name tutorial_${id})
    build_g(${generator_target} tutorials/tutorial_${id}.cpp ${generated_obj})
    build_w(${tutorial_name} "${generated_obj}" tutorials/wrapper_tutorial_${id}.cpp tutorials/wrapper_tutorial_${id}.h)
    add_custom_target(run_tutorial_${id} COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    add_custom_command(TARGET tutorials COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
endfunction()

foreach(t ${TIRAMISU_TUTORIALS})
    new_tutorial(${t})
endforeach()
