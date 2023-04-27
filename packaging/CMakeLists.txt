include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


set(Tiramisu_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tiramisu"
    CACHE STRING "Path to Halide CMake files")


configure_package_config_file(common/tiramisuConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/tiramisuConfig.cmake
							  INSTALL_DESTINATION  ${Tiramisu_INSTALL_CMAKEDIR})


## SETUP INSTALLS:

install(TARGETS tiramisu
EXPORT tiramisu_Targets
LIBRARY DESTINATION lib
INCLUDES DESTINATION include
PUBLIC_HEADER DESTINATION include/tiramisu
)

if (USE_AUTO_SCHEDULER)
   install(TARGETS tiramisu_auto_scheduler
   EXPORT tiramisu_Targets
   LIBRARY DESTINATION lib
   INCLUDES DESTINATION include
   PUBLIC_HEADER DESTINATION include/tiramisu/autoscheduler)
endif()


if (WITH_PYTHON_BINDINGS)
  install(FILES ${CMAKE_SOURCE_DIR}/python_bindings/tiramisu.pyi ${CMAKE_SOURCE_DIR}/python_bindings/__init__.py
  DESTINATION "${Tiramisu_INSTALL_PYTHONDIR}/tiramisu")
  install(TARGETS Tiramisu_Python
  EXPORT tiramisu_Targets
  LIBRARY DESTINATION "${Tiramisu_INSTALL_PYTHONDIR}/tiramisu")
endif()


install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tiramisuConfig.cmake
	${CMAKE_SOURCE_DIR}/cmake/TiramisuGeneratorHelpers.cmake
        DESTINATION ${Tiramisu_INSTALL_CMAKEDIR})
install(EXPORT tiramisu_Targets
	FILE tiramisuTargets.cmake
	DESTINATION ${Tiramisu_INSTALL_CMAKEDIR})
