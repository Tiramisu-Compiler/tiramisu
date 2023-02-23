find_library(HALIDE_LIB Halide::Halide)

get_filename_component(HALIDE_LIB_DIRECTORY ${HALIDE_LIB} DIRECTORY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHALIDE_BUILD=\"${HALIDE_LIB_DIRECTORY}\"")

set(SOURCES
  PyTiramisu.cpp
  PyExpr.cpp
  PyType.cpp
  PyBuffer.cpp
  PyInit.cpp
  PyVar.cpp
  PyComputation.cpp
  PyConstant.cpp
  PyCodegen.cpp
  PyInput.cpp
  PyFunction.cpp
  )


pybind11_add_module(Tiramisu_Python MODULE ${SOURCES})
set_target_properties(Tiramisu_Python
  PROPERTIES
  LIBRARY_OUTPUT_NAME tiramisu
  EXPORT_NAME Python)
target_link_libraries(Tiramisu_Python PRIVATE tiramisu) #Tiramisu_Python needs Tiramisu


