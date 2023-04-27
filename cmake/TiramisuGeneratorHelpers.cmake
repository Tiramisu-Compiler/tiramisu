
# Export Tiramisu Versions...
function(_Halide_try_load_generators)
    # Don't repeatedly run the search for the tools package.
    if (NOT DEFINED ${ARG_PACKAGE_NAME}_FOUND)
        # Some toolchains, like Emscripten, try to disable finding packages
        # outside their sysroots, but we always want to find the native
        # generators. Setting CMAKE_FIND_ROOT_PATH_BOTH here overrides
        # the toolchain search preference. This is okay since a user can
        # always override this call by setting ${ARG_PACKAGE_NAME}_ROOT.
        find_package(${ARG_PACKAGE_NAME} QUIET
                     CMAKE_FIND_ROOT_PATH_BOTH)

        # Communicate found information to the caller
        set(${ARG_PACKAGE_NAME}_FOUND "${${ARG_PACKAGE_NAME}_FOUND}" PARENT_SCOPE)

        if (NOT ${ARG_PACKAGE_NAME}_FOUND)
            message(WARNING
                    "'${ARG_PACKAGE_NAME}' was not found and it looks like you "
                    "are cross-compiling without an emulator. This is likely to "
                    "fail. Please set -D${ARG_PACKAGE_NAME}_ROOT=... at the CMake "
                    "command line to the build directory of a host-built ${PROJECT_NAME}.")
        endif ()
    endif ()
endfunction()

function(add_tiramisu_generator TARGET)
    set(options "")
    set(oneValueArgs PACKAGE_NAME PACKAGE_NAMESPACE EXPORT_FILE)
    set(multiValueArgs SOURCES LINK_LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT ARG_PACKAGE_NAME)
        set(ARG_PACKAGE_NAME "${PROJECT_NAME}-tiramisu_generators")
    endif ()

     if (NOT ARG_PACKAGE_NAMESPACE)
        set(ARG_PACKAGE_NAMESPACE "${PROJECT_NAME}::tiramisu_generators::")
    endif ()

    if (NOT ARG_EXPORT_FILE)
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cmake")
        set(ARG_EXPORT_FILE "${PROJECT_BINARY_DIR}/cmake/${ARG_PACKAGE_NAME}Config.cmake")
    endif ()

    _Halide_try_load_generators()
    

    set(${ARG_PACKAGE_NAME}_FOUND "${${ARG_PACKAGE_NAME}_FOUND}" PARENT_SCOPE)

    set(gen "${ARG_PACKAGE_NAMESPACE}${TARGET}")
    if (NOT TARGET "${gen}")
        if (NOT TARGET "${ARG_PACKAGE_NAME}")
            add_custom_target("${ARG_PACKAGE_NAME}")
        endif ()
    endif()

     find_package(tiramisu REQUIRED)
     add_executable(${TARGET} ${ARG_SOURCES})
     add_executable(${gen} ALIAS ${TARGET})
     target_link_libraries(${TARGET} tiramisu Halide::Halide ${ARG_LINK_LIBRARIES})
     add_dependencies("${ARG_PACKAGE_NAME}" ${TARGET})
     export(TARGETS ${TARGET}
            NAMESPACE ${ARG_PACKAGE_NAMESPACE}
            APPEND FILE "${ARG_EXPORT_FILE}")
endfunction()

function(add_tiramisu_library TARGET)
    set(options "")
    set(oneValueArgs FROM GENERATOR FUNCTION_NAME NAMESPACE)
    set(multiValueArgs PARAMS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"  ${ARGN})

    if (NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(AUTHOR_WARNING "Arguments to add_tiramisu_library were not recognized: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()

    if (NOT ARG_FROM)
        message(FATAL_ERROR "Missing FROM argument specifying a Halide generator target")
    endif ()

    if (NOT ARG_GENERATOR)
        set(ARG_GENERATOR "${TARGET}")
    endif ()

    if (NOT ARG_FUNCTION_NAME)
        set(ARG_FUNCTION_NAME "${TARGET}")
    endif ()

    if (ARG_NAMESPACE)
        set(ARG_FUNCTION_NAME "${ARG_NAMESPACE}::${ARG_FUNCTION_NAME}")
    endif ()

    

    set(GENERATOR_CMD "${ARG_FROM}")
    set(GENERATOR_CMD_DEPS ${ARG_FROM})
    list(APPEND generator_output_files "${ARG_FUNCTION_NAME}.o" "${ARG_FUNCTION_NAME}.o.h")

    add_custom_command(OUTPUT ${generator_output_files}
                       COMMAND ${GENERATOR_CMD} ${ARG_PARAMS}
		       DEPENDS ${GENERATOR_CMD_DEPS}
		       VERBATIM)
    list(TRANSFORM generator_output_files PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
    add_custom_target("${TARGET}.update" ALL DEPENDS ${generator_output_files})
    add_library("${TARGET}" STATIC "${ARG_FUNCTION_NAME}.o")
    set_target_properties("${TARGET}" PROPERTIES
    			  POSITION_INDEPENDENT_CODE ON
                          LINKER_LANGUAGE CXX)

    add_dependencies("${TARGET}" "${TARGET}.update")
    target_include_directories("${TARGET}" INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
    target_link_libraries("${TARGET}" Halide::Runtime tiramisu)
endfunction()


function(add_tiramisu_test_library TARGET)
    set(options "")
    set(oneValueArgs FROM GENERATOR FUNCTION_NAME NAMESPACE)
    set(multiValueArgs PARAMS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"  ${ARGN})

    if (NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(AUTHOR_WARNING "Arguments to add_tiramisu_library were not recognized: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()

    if (NOT ARG_FROM)
        message(FATAL_ERROR "Missing FROM argument specifying a Halide generator target")
    endif ()

    if (NOT ARG_GENERATOR)
        set(ARG_GENERATOR "${TARGET}")
    endif ()

    if (NOT ARG_FUNCTION_NAME)
        set(ARG_FUNCTION_NAME "${TARGET}")
    endif ()

    if (ARG_NAMESPACE)
        set(ARG_FUNCTION_NAME "${ARG_NAMESPACE}::${ARG_FUNCTION_NAME}")
    endif ()

    

    set(GENERATOR_CMD "${ARG_FROM}")
    set(GENERATOR_CMD_DEPS ${ARG_FROM})
    list(APPEND generator_output_files "${ARG_FUNCTION_NAME}.o" "${ARG_FUNCTION_NAME}.o.h")

    add_custom_command(OUTPUT ${generator_output_files}
                       COMMAND ${GENERATOR_CMD} ${ARG_PARAMS}
		       DEPENDS ${GENERATOR_CMD_DEPS}
		       VERBATIM)
    list(TRANSFORM generator_output_files PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
    add_custom_target("${TARGET}.update" ALL DEPENDS ${generator_output_files})
    add_library("${TARGET}" STATIC "${ARG_FUNCTION_NAME}.o")
    set_target_properties("${TARGET}" PROPERTIES
    			  POSITION_INDEPENDENT_CODE ON
                          LINKER_LANGUAGE CXX
			  EXCLUDE_FROM_ALL true)

    add_dependencies("${TARGET}" "${TARGET}.update")
    target_include_directories("${TARGET}" INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
    target_link_libraries("${TARGET}" Halide::Runtime tiramisu)
endfunction()



# Exported Functions for use downstream.
# Gen
# Use Gen 
