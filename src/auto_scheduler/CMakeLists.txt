# set(OBJS_AUTO_SCHEDULER auto_scheduler optimization_info dnn_accesses ast evaluator schedules_generator search_method)

set(AUTO_SOURCES
tiramisu_ast.cpp
tiramisu_auto_scheduler.cpp
tiramisu_dnn_accesses.cpp
tiramisu_evaluator.cpp
tiramisu_optimization_info.cpp
tiramisu_schedules_generator.cpp
tiramisu_search_method.cpp
)

set(AUTO_HEADERS
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/auto_scheduler.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/utils.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/optimization_info.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/dnn_accesses.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/ast.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/evaluator.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/schedules_generator.h
        ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler/search_method.h
)

add_library(tiramisu_auto_scheduler SHARED ${AUTO_SOURCES})
target_link_libraries(tiramisu_auto_scheduler tiramisu Halide::Halide Halide::Runtime Halide::Tools)
target_link_libraries(tiramisu_auto_scheduler ${LINK_FLAGS}))
target_link_libraries(tiramisu_auto_scheduler ${LINK_FLAGS})
target_include_directories(tiramisu_auto_scheduler PUBLIC "${ISL_INCLUDE_DIRECTORY}")
target_include_directories(tiramisu_auto_scheduler PUBLIC ${CMAKE_SOURCE_DIR}/include/tiramisu/auto_scheduler)

set_target_properties(tiramisu_auto_scheduler
  PROPERTIES
  LIBRARY_OUTPUT_NAME tiramisu_auto_scheduler
  EXPORT_NAME tiramisu_auto_scheduler
  PUBLIC_HEADER "${AUTO_HEADERS}"
)


