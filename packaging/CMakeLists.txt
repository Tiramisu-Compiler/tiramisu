include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


set(Tiramisu_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tiramisu"
    CACHE STRING "Path to Halide CMake files")


configure_file(common/tiramisuConfig.cmake tiramisuConfig.cmake @ONLY)


install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tiramisuConfig.cmake
        DESTINATION ${Tiramisu_INSTALL_CMAKEDIR})
