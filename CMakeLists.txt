cmake_minimum_required(VERSION 3.22)
project(Tiramisu)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE "ON")
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include(CMakeDependentOption)
include(CheckCXXSymbolExists)

# Make our custom helpers available throughout the project via include().
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(TiramisuGeneratorHelpers)

include(configure.cmake)
if (DEBUG_LEVEL GREATER_EQUAL 1)
   set(ENABLE_DEBUG TRUE)
else()
   set(ENABLE_DEBUG FALSE)
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(PROJECT_BUILD ${CMAKE_BINARY_DIR})
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

### FIND EXTERNAL DEPS ###
if (${USE_GPU})
    find_package(CUDA REQUIRED)
    add_definitions(-DNVCC_PATH="${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
    add_library(cuda_wrapper STATIC "src/tiramisu_cuda_wrappers.cpp")
    target_link_libraries(cuda_wrapper ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    target_include_directories(cuda_wrapper PUBLIC ${CUDA_INCLUDE_DIRS})
    if (${USE_CUDNN})
        find_library(CUDNN_LIBRARIES cudnn PATHS ${CUDNN_LOCATION}/lib64 NO_DEFAULT_PATH)
        set(CUDNN_INCLUDE_DIRECTORY ${CUDNN_LOCATION}/include)
    endif()
endif()

if (${USE_MKL_WRAPPERS})
    add_library(mkl_wrapper STATIC "src/tiramisu_mkl_wrappers.cpp")
    target_include_directories(mkl_wrapper PUBLIC ${MKL_PREFIX}/include)
endif()

macro(init_tags)
    set(is_gpu false)
    set(is_mpi false)
    set(NUM_MPI_RANKS 0)
    set(is_cudnn false)
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
        elseif (${tag} STREQUAL cudnn)
            set(is_cudnn true)
        else()
            message(WARNING "Tag ${tag} not recognized")
        endif()
    endforeach()
endmacro()



# Check that MPI variables are set correctly if we need it
if (${USE_MPI})
    if (MPI_BUILD_DIR STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_BUILD_DIR is empty.")
    elseif (MPI_NODES STREQUAL "")
        message(FATAL_ERROR "USE_MPI==true but MPI_NODES is empty.")
    endif()
endif()

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)  # or newer
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)
find_library(ISLLib isl PATHS ${ISL_LIB_DIRECTORY} NO_DEFAULT_PATH)

if (${USE_MPI})
    set(CMAKE_CXX_COMPILER "${MPI_BUILD_DIR}/bin/mpicxx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_MPI")
    link_directories(${MPI_BUILD_DIR}/lib)
    include_directories(${MPI_BUILD_DIR}/include)
endif()




set(LINK_FLAGS "-ldl -lpthread")
if(${USE_MPI})
    set(LINK_FLAGS "${LINK_FLAGS} -lmpi")
endif()


if(${ENABLE_DEBUG})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_DEBUG -DDEBUG_LEVEL=${DEBUG_LEVEL}")
endif()

set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)


### Create targets ###

add_subdirectory(src)

if (WITH_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings enabled")
  add_subdirectory(python_bindings)
  target_include_directories(Tiramisu_Python PRIVATE "include")
endif()




### BROKEN: SETUP TESTS/BENCHMARKS/ETC ###


if (APPLE)
    set(LIB_SUF dylib)
else ()
    set(LIB_SUF so)
endif ()
if (WITH_TESTS)
function(new_test descriptor)
    parse_descriptor(${descriptor})
    set(generator_target test_${id}_fct_generator)
    set_obj(${PROJECT_BUILD}/generated_fct_test_${id}.o)
    set(test_name test_${id})
    build_g(${generator_target} tests/test_${id}.cpp "${obj}")
    build_w(${test_name} "${obj}" tests/wrapper_test_${id}.cpp tests/wrapper_test_${id}.h)
    add_test(NAME ${id}_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${test_name})
    if (NOT ${is_mpi})
        add_test(NAME ${id} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI}) # This is an MPI test (sanity check that we want to use MPI though)
        # configure the options so files are copied on the fly as necessary.
        add_test(NAME ${id} COMMAND ${MPI_BUILD_DIR}/bin/mpirun -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} --map-by node --oversubscribe --wdir /tmp/ --preload-files ${PROJECT_BUILD}/libtiramisu.${LIB_SUF},${PROJECT_DIR}/3rdParty/isl/.libs/libisl.${LIB_SUF} --preload-binary ${PROJECT_BUILD}/${test_name} WORKING_DIRECTORY ${PROJECT_DIR})
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
endif() #WITH_TESTS

if (WITH_BENCHMARKS)
add_custom_target(benchmarks)

function(new_benchmark descriptor)
    parse_descriptor(${descriptor})
    set(tiramisu_generator_target bench_tiramisu_${id}_generator)
    set(halide_generator_target   bench_halide_${id}_generator)
    set_obj("${PROJECT_BUILD}/generated_fct_${id}.o")
    set(generated_obj_halide   ${PROJECT_BUILD}/generated_fct_${id}_ref.o)
    set(bench_name bench_${id})
    build_g(${tiramisu_generator_target} benchmarks/halide/${id}_tiramisu.cpp "${obj}")
    build_halide_g(${halide_generator_target} benchmarks/halide/${id}_ref.cpp ${generated_obj_halide})
    build_w(${bench_name} "${obj};${generated_obj_halide}" benchmarks/halide/wrapper_${id}.cpp benchmarks/halide/wrapper_${id}.h)
    if (NOT ${is_mpi})
        add_custom_target(run_benchmark_${id} COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
        add_custom_command(TARGET benchmarks COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI})
        # configure the options so files are copied on the fly as necessary.
        add_custom_target(run_benchmark_${id} COMMAND ${MPI_BUILD_DIR}/bin/mpirun -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} --map-by node --oversubscribe -wdir ${PROJECT_DIR} --preload-files ${PROJECT_BUILD}/libtiramisu.${LIB_SUF},${PROJECT_DIR}/3rdParty/isl/.libs/libisl.${LIB_SUF} --preload-binary ${PROJECT_BUILD}/${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()
    add_dependencies(run_benchmark_${id} ${bench_name})
endfunction()

foreach(b ${TIRAMISU_BENCHMARKS})
    new_benchmark(${b})
endforeach()


add_custom_target(dist_benchmarks)

function(new_dist_benchmark descriptor)
    parse_descriptor(${descriptor})
    set(tiramisu_generator_target dist_bench_tiramisu_${id}_generator)
    set(ref_generator_target   dist_bench_ref_${id}_generator)
    set_obj("${PROJECT_BUILD}/generated_fct_${id}.o")
    set(generated_obj_ref   ${PROJECT_BUILD}/generated_fct_${id}_ref.o)
    set(bench_name bench_${id})
    build_g(${tiramisu_generator_target} benchmarks/automatic_comm/${id}_tiramisu.cpp "${obj}")
    build_g(${ref_generator_target} benchmarks/automatic_comm/${id}_ref.cpp ${generated_obj_ref})
    build_w(${bench_name} "${obj};${generated_obj_ref}" benchmarks/automatic_comm/wrapper_${id}.cpp benchmarks/automatic_comm/wrapper_${id}.h)
    if (NOT ${is_mpi})
        add_custom_target(run_dist_benchmark_${id} COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
        add_custom_command(TARGET dist_benchmarks COMMAND ${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI})
        # configure the options so files are copied on the fly as necessary.
        add_custom_target(run_dist_benchmark_${id} COMMAND ${MPI_BUILD_DIR}/bin/mpirun -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} --map-by node --oversubscribe -wdir ${PROJECT_DIR} --preload-files ${PROJECT_BUILD}/libtiramisu.${LIB_SUF},${PROJECT_DIR}/3rdParty/isl/.libs/libisl.${LIB_SUF} --preload-binary ${PROJECT_BUILD}/${bench_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()
    add_dependencies(run_dist_benchmark_${id} ${bench_name})
endfunction()

foreach(b ${TIRAMISU_DIST_BENCHMARKS})
    new_dist_benchmark(${b})
endforeach()

# Individual benchmarks are moved to benchmarks/CMakeLists.txt to reduce clutter
add_subdirectory(benchmarks)

endif() #WITH_BENCHMAKRS

if (WITH_DOCS)
add_custom_target(doc DEPENDS ${PROJECT_DIR}/utils/doc_generation/Doxyfile)
add_custom_command(TARGET doc COMMAND doxygen utils/doc_generation/Doxyfile WORKING_DIRECTORY ${PROJECT_DIR})
endif() #WITH_DOCS

if (WITH_TUTORIALS)
add_custom_target(tutorials)


function(new_developers_tutorial descriptor)
    parse_descriptor(${descriptor})
    set(generator_target developers_tutorial_${id}_fct_generator)
    set_obj("${PROJECT_BUILD}/generated_fct_developers_tutorial_${id}.o")
    set(tutorial_name tutorial_developers_${id})
    build_g(${generator_target} tutorials/developers/tutorial_${id}/tutorial_${id}.cpp "${obj}")
    build_w(${tutorial_name} "${obj}" tutorials/developers/tutorial_${id}/wrapper_tutorial_${id}.cpp tutorials/developers/tutorial_${id}/wrapper_tutorial_${id}.h)
    if (NOT ${is_mpi})
        add_custom_target(run_developers_tutorial_${id} COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
        add_custom_command(TARGET tutorials COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI}) # This is an MPI test (sanity check that we want to use MPI though)
        # configure the options so files are copied on the fly as necessary.
        add_custom_target(run_developers_tutorial_${id} COMMAND ${MPI_BUILD_DIR}/bin/mpirun -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} --map-by node --oversubscribe --wdir /tmp/ --preload-files ${PROJECT_BUILD}/libtiramisu.${LIB_SUF},${PROJECT_DIR}/3rdParty/isl/.libs/libisl.${LIB_SUF} --preload-binary ${PROJECT_BUILD}/${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()
    add_dependencies(run_developers_tutorial_${id} ${tutorial_name})
endfunction()

function(new_users_tutorial descriptor)
    parse_descriptor(${descriptor})
    set(generator_target users_tutorial_${id}_fct_generator)
    set_obj("${PROJECT_BUILD}/generated_fct_users_tutorial_${id}.o")
    set(tutorial_name tutorial_users_${id})
    build_g(${generator_target} tutorials/users/tutorial_${id}/tutorial_${id}.cpp "${obj}")
    build_w(${tutorial_name} "${obj}" tutorials/users/tutorial_${id}/wrapper_tutorial_${id}.cpp tutorials/users/tutorial_${id}/wrapper_tutorial_${id}.h)
    if (NOT ${is_mpi})
        add_custom_target(run_users_tutorial_${id} COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
        add_custom_command(TARGET tutorials COMMAND ${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    elseif (${USE_MPI}) # This is an MPI test (sanity check that we want to use MPI though)
        # configure the options so files are copied on the fly as necessary.
        add_custom_target(run_users_tutorial_${id} COMMAND ${MPI_BUILD_DIR}/bin/mpirun -x LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/tmp/ -np ${NUM_MPI_RANKS} -host ${MPI_NODES} --map-by node --oversubscribe --wdir /tmp/ --preload-files ${PROJECT_BUILD}/libtiramisu.${LIB_SUF},${PROJECT_DIR}/3rdParty/isl/.libs/libisl.${LIB_SUF} --preload-binary ${PROJECT_BUILD}/${tutorial_name} WORKING_DIRECTORY ${PROJECT_DIR})
    endif()
    add_dependencies(run_users_tutorial_${id} ${tutorial_name})
endfunction()



   foreach(t ${TIRAMISU_DEVELOPERS_TUTORIALS})
   	     new_developers_tutorial(${t})
   endforeach()
   foreach(t ${TIRAMISU_USERS_TUTORIALS})
   	      new_users_tutorial(${t})
   endforeach()
endif() #WITH_TUTORIALS

add_subdirectory(tests)
add_subdirectory(packaging)
