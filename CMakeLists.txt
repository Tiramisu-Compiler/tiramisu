cmake_minimum_required(VERSION 3.5)
project(tiramisu)

enable_testing()

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE "ON")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")


# Add CMake header files
set(HEADER_FILES
        include/tiramisu/cuda_ast.h
        include/tiramisu/core.h
        include/tiramisu/debug.h
        include/tiramisu/utils.h
        include/tiramisu/type.h
        include/tiramisu/expr.h
        include/tiramisu/mpi_comm.h
        include/tiramisu/externs.h
        )

# Add CMake cpp files
set(OBJS expr core codegen_halide codegen_c debug utils codegen_halide_lowering codegen_from_halide mpi codegen_cuda externs)


### CMAKE FILE INTERNALS ###

include(configure.cmake)

if (${USE_GPU})
    find_package(CUDA REQUIRED)
    set(GPU_ENV_EXEC "${CMAKE_COMMAND} -E env NVCC_BIN_DIR=${CUDA_TOOLKIT_ROOT_DIR}/bin/")
    add_library(cuda_wrapper STATIC "src/tiramisu_cuda_wrappers.cpp")
    target_link_libraries(cuda_wrapper ${CUDA_LIBRARIES})
    target_include_directories(cuda_wrapper PUBLIC ${CUDA_INCLUDE_DIRS})
endif()


FILE(STRINGS tests/test_list.txt TIRAMISU_TESTS_RAW)
FILE(STRINGS benchmarks/benchmark_list.txt TIRAMISU_BENCHMARKS_RAW)
FILE(STRINGS tutorials/developers_tutorial_list.txt TIRAMISU_DEVELOPERS_TUTORIALS_RAW)
FILE(STRINGS tutorials/users_tutorial_list.txt TIRAMISU_USERS_TUTORIALS_RAW)


macro(init_tags)
    set(is_gpu false)
    set(is_mpi false)
    set(NUM_MPI_RANKS 0)
endmacro()

macro(parse_tags tags)
    init_tags() 
    foreach(tag ${tags})
        if (${tag} STREQUAL gpu)
            set(is_gpu true)
        elseif (${tag} STREQUAL mpi)
            set(is_mpi true)
        elseif (${tag} GREATER -1) # check if it is a number
            set(NUM_MPI_RANKS ${tag})
        else()
            message(WARNING "Tag ${tag} not recognized")
        endif()
    endforeach()
endmacro()

# transforms the list of the format (program_name/program_name[tag0,tag1,..,tagn]) into a list
# containing a list of "program_name property0 property1 property2 .. propertyn"
# the list can contain as many tags as necessary
# the number of properties is fixed and is in the order given below
# current tags are : gpu, mpi.
# current properties are : is_gpu(boolean), is_mpi(boolean)
function (parse_list raw_list list_name)
    set(result)
    foreach(element ${raw_list})
        set(tmp_result)
        init_tags()
        string(REGEX MATCH "^([a-zA-Z_0-9]+)(\\[([a-zA-Z0-9_]+(,[a-zA-Z0-9_]+)*)\\])?$" the_match "${element}")
        if(NOT ${CMAKE_MATCH_COUNT})
            message(WARNING "Could not parse: ${element}")
            continue()
        elseif(${CMAKE_MATCH_COUNT} GREATER 1)
            STRING(REPLACE "," ";" tags ${CMAKE_MATCH_3})
            parse_tags("${tags}")
        endif()
        set(name ${CMAKE_MATCH_1})
        # The list should always have four elements (name <USE_GPU> <USE_MPI> <NUM_MPI_RANKS>)
        list(APPEND tmp_result "${name}")
        # Check if we need to use the GPU
        if (NOT ${is_gpu} OR NOT "${USE_GPU}")
          if (${is_gpu} AND NOT "${USE_GPU}")
            message(STATUS "skipping gpu tagged ${name} because USE_GPU is not set to true")
          endif ()
            list(APPEND tmp_result "false")
        else () # Both are true
            list(APPEND tmp_result "true")
        endif()
        # Check if we need to use MPI
        if (NOT ${is_mpi} OR NOT "${USE_MPI}")
          if (${is_mpi} AND NOT "${USE_MPI}")
            message(STATUS "skipping mpi tagged ${name} because USE_MPI is not set to true")
          endif()
            list(APPEND tmp_result "false")
            list(APPEND tmp_result "0")
        else ()
            list(APPEND tmp_result "true")
            list(APPEND tmp_result "${NUM_MPI_RANKS}")
        endif()
        string(REPLACE ";" " " tmp_result "${tmp_result}")
        list(APPEND result "${tmp_result}")
    endforeach()
    set("${list_name}" "${result}" PARENT_SCOPE)
endfunction()


parse_list("${TIRAMISU_BENCHMARKS_RAW}" TIRAMISU_BENCHMARKS)
parse_list("${TIRAMISU_TESTS_RAW}" TIRAMISU_TESTS)
parse_list("${TIRAMISU_DEVELOPERS_TUTORIALS_RAW}" TIRAMISU_DEVELOPERS_TUTORIALS)
parse_list("${TIRAMISU_USERS_TUTORIALS_RAW}" TIRAMISU_USERS_TUTORIALS)


# Check that MPI variables are set correctly if we need it
if (${USE_MPI})
    if (MPI_INCLUDE_DIR STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_INCLUDE_DIR is empty.")
    elseif (MPI_LIB_DIR STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_LIB_DIR is empty.")
    elseif (MPI_LIB_FLAGS STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_LIB_FLAGS is empty.")
    elseif (MPI_NODES STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_NODES is empty.")
    endif()
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-sign-compare -fno-rtti -DHALIDE_NO_JPEG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories("include"
        "${ISL_INCLUDE_DIRECTORY}"
        "${HALIDE_SOURCE_DIRECTORY}/include" "${HALIDE_SOURCE_DIRECTORY}/tools"
        "build")

if (${USE_MPI})
    set(CMAKE_CXX_COMPILER "mpicxx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MPI")
    link_directories(${MPI_LIB_DIR})
    include_directories(${MPI_INCLUDE_DIRS})
else ()
    set(CMAKE_CXX_COMPILER "g++")
endif()


find_library(HalideLib Halide PATHS ${HALIDE_LIB_DIRECTORY} NO_DEFAULT_PATH)
find_library(ISLLib isl PATHS ${ISL_LIB_DIRECTORY} NO_DEFAULT_PATH)

execute_process(COMMAND libpng-config --ldflags OUTPUT_VARIABLE LIBPNG_FLAGS)

# Require LLVM 5.0 or greater to keep in line with Halide
execute_process(COMMAND ${LLVM_CONFIG_BIN}/llvm-config --version OUTPUT_VARIABLE LLVM_VERSION)
string(STRIP ${LLVM_VERSION} LLVM_VERSION)
if (${LLVM_VERSION} VERSION_GREATER 5.0)
    execute_process(COMMAND ${LLVM_CONFIG_BIN}/llvm-config --ignore-libllvm --system-libs OUTPUT_VARIABLE LLVM_FLAGS)
else ()
    message(FATAL_ERROR "tiramisu requires LLVM version >= 5.0")
endif()

string(STRIP ${LLVM_FLAGS} LLVM_FLAGS)
string(STRIP ${LIBPNG_FLAGS} LIBPNG_FLAGS)

set(LINK_FLAGS "-ldl -lpthread")
if (${USE_MPI})
    set(LINK_FLAGS "${LINK_FLAGS} ${LLVM_FLAGS} ${LIBPNG_FLAGS} ${MPI_LIB_FLAGS}")
else()
    set(LINK_FLAGS "${LINK_FLAGS} ${LLVM_FLAGS} ${LIBPNG_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/build)

set(T_CODE "")

foreach (obj ${OBJS})
    set(T_CODE ${T_CODE} src/tiramisu_${obj}.cpp)
endforeach ()

add_library(tiramisu SHARED ${T_CODE})
target_link_libraries(tiramisu ${HalideLib} ${ISLLib})
target_link_libraries(tiramisu ${LINK_FLAGS})

function(link_tags name)
    if (${is_gpu})
      if (${USE_GPU})
        target_link_libraries(${name} cuda_wrapper)
      endif()
    endif()
endfunction()

function(build_e name generator)
    add_executable(${name} ${generator} ${HEADER_FILES})
    target_link_libraries(${name} tiramisu ${HalideLib} ${ISLLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    link_tags(${name})
endfunction()

function(build_g name generator result)
    add_executable(${name} ${generator} ${HEADER_FILES})
    target_link_libraries(${name} tiramisu ${HalideLib} ${ISLLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    link_tags(${name})
    if (NOT result STREQUAL "")
        set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${result}")
        add_custom_command(OUTPUT ${result} COMMAND ${name} WORKING_DIRECTORY ${PROJECT_DIR} DEPENDS ${name})
    endif()
endfunction()

function(build_halide_g name generator result)
    add_executable(${name} ${generator})
    target_link_libraries(${name} ${HalideLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    if (NOT result STREQUAL "")
        set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${result})
        add_custom_command(OUTPUT ${result} COMMAND ${name} WORKING_DIRECTORY ${PROJECT_DIR} DEPENDS ${NAME})
    endif()
endfunction()

function(build_w name objs wrapper header)
    add_executable(${name} ${wrapper} ${objs} ${header})
    target_link_libraries(${name} tiramisu ${HalideLib} ${ISLLib})
    target_link_libraries(${name} ${LINK_FLAGS})
    link_tags(${name})
endfunction()

macro(set_obj inp_obj)
    if (${is_gpu})
        set(obj "${inp_obj};${inp_obj}_gpu.o;${inp_obj}_cpu.o")
    else()
        set(obj "${inp_obj}")
    endif()
endmacro()

function(parse_descriptor descriptor)
    string(REPLACE " " ";" descriptor ${descriptor})
    list(GET descriptor 0 id)
    set(id ${id} PARENT_SCOPE)
    list(GET descriptor 1 is_gpu)
    set(is_gpu ${is_gpu} PARENT_SCOPE)
    list(GET descriptor 2 is_mpi)
    set(is_mpi ${is_mpi} PARENT_SCOPE)
    list(GET descriptor 3 NUM_MPI_RANKS)
    set(NUM_MPI_RANKS ${NUM_MPI_RANKS} PARENT_SCOPE)
endfunction()

function(new_test descriptor)
    parse_descriptor(${descriptor})
    set(generator_target test_${id}_fct_generator)
    set_obj(${PROJECT_DIR}/build/generated_fct_test_${id}.o)
    set(test_name test_${id})
    build_g(${generator_target} tests/test_${id}.cpp "${obj}")
    build_w(${test_name} "${obj}" tests/wrapper_test_${id}.cpp tests/wrapper_test_${id}.h)
    add_test(NAME ${id}_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${test_name})
    if (NOT ${is_mpi})
        add_test(NAME ${id} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI}) # This is an MPI test (sanity check that we want to use MPI though)
        # configure the options so files are copied on the fly as necessary. 
        find_program(MPI_RUN_COMMAND NAMES mpirun)        
        add_test(NAME ${id} COMMAND ${MPI_RUN_COMMAND} -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} -bynode --preload-files-dest-dir /tmp/ --preload-files build/libtiramisu.so,3rdParty/isl/.libs/libisl.so.19 --preload-binary build/${test_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()
    set_tests_properties(${id} PROPERTIES DEPENDS ${id}_build)
endfunction()

build_g(test_global tests/test_global.cpp "")
add_test(NAME global_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target test_global)
add_test(NAME global COMMAND test_global WORKING_DIRECTORY ${PROJECT_DIR})
set_tests_properties(global PROPERTIES DEPENDS global_build)
foreach(t ${TIRAMISU_TESTS})
    new_test(${t})
endforeach()

add_custom_target(benchmarks)

function(new_benchmark descriptor)
    parse_descriptor(${descriptor})
    set(tiramisu_generator_target bench_tiramisu_${id}_generator)
    set(halide_generator_target   bench_halide_${id}_generator)
    set_obj("${PROJECT_DIR}/build/generated_fct_${id}.o")
    set(generated_obj_halide   ${PROJECT_DIR}/build/generated_fct_${id}_ref.o)
    set(bench_name bench_${id})
    build_g(${tiramisu_generator_target} benchmarks/halide/${id}_tiramisu.cpp "${obj}")
    build_halide_g(${halide_generator_target} benchmarks/halide/${id}_ref.cpp ${generated_obj_halide})
    build_w(${bench_name} "${obj};${generated_obj_halide}" benchmarks/halide/wrapper_${id}.cpp benchmarks/halide/wrapper_${id}.h)
    add_custom_target(run_benchmark_${id} COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    add_custom_command(TARGET benchmarks COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
endfunction()

foreach(b ${TIRAMISU_BENCHMARKS})
    new_benchmark(${b})
endforeach()

add_custom_target(doc DEPENDS ${PROJECT_DIR}/utils/doc_generation/Doxyfile)
add_custom_command(TARGET doc COMMAND doxygen utils/doc_generation/Doxyfile WORKING_DIRECTORY ${PROJECT_DIR})

add_custom_target(tutorials)

function(new_developers_tutorial descriptor)
    parse_descriptor(${descriptor})
    set(generator_target developers_tutorial_${id}_fct_generator)
    set_obj("${PROJECT_DIR}/build/generated_fct_developers_tutorial_${id}.o")
    set(tutorial_name tutorial_${id})
    build_g(${generator_target} tutorials/developers/tutorial_${id}/tutorial_${id}.cpp "${obj}")
    build_w(${tutorial_name} "${obj}" tutorials/developers/tutorial_${id}/wrapper_tutorial_${id}.cpp tutorials/developers/tutorial_${id}/wrapper_tutorial_${id}.h)
    add_custom_target(run_developers_tutorial_${id} COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    if (NOT ${is_mpi})
        add_custom_command(TARGET tutorials COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
        add_custom_command(TARGET run_developers_tutorial_${id} COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI}) # This is an MPI test (sanity check that we want to use MPI though)
        # configure the options so files are copied on the fly as necessary. 
        find_program(MPI_RUN_COMMAND NAMES mpirun)        
        add_custom_command(TARGET run_developers_tutorial_${id} COMMAND ${MPI_RUN_COMMAND} -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} -bynode --preload-files-dest-dir /tmp/ --preload-files build/libtiramisu.so,3rdParty/isl/.libs/libisl.so.19 --preload-binary build/${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()

endfunction()

function(new_users_tutorial descriptor)
    parse_descriptor(${descriptor})
    set(generator_target users_tutorial_${id}_fct_generator)
    set_obj("${PROJECT_DIR}/build/generated_fct_users_tutorial_${id}.o")
    set(tutorial_name tutorial_${id})
    build_e(${generator_target} tutorials/users/tutorial_${id}/tutorial_${id}.cpp)
    add_custom_target(run_users_tutorial_${id} COMMAND ${generator_target} WORKING_DIRECTORY ${PROJECT_DIR})
    add_custom_command(TARGET tutorials COMMAND ${generator_target} WORKING_DIRECTORY ${PROJECT_DIR})
endfunction()


foreach(t ${TIRAMISU_DEVELOPERS_TUTORIALS})
    new_developers_tutorial(${t})
endforeach()

foreach(t ${TIRAMISU_USERS_TUTORIALS})
    new_users_tutorial(${t})
endforeach()

