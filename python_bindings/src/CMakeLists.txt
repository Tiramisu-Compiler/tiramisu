set(SOURCES
  PyTiramisu.cpp
  PyExpr.cpp
  PyType.cpp
  PyBuffer.cpp
  PyInit.cpp
  PyConstant.cpp
  PyVar.cpp
  PyComputation.cpp
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
#target_include_directories(Tiramisu_Python PRIVATE "PyTiramisu.h" "PyExpr.h")


