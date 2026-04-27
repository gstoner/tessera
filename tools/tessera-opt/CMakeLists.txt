
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  TesseraIR
  TesseraPasses
  tessera_solvers_passes
  tessera_linalg_solver_passes
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIROptLib
)

target_include_directories(tessera-opt PRIVATE
  ${CMAKE_SOURCE_DIR}/src/solvers/core/passes
  ${CMAKE_SOURCE_DIR}/src/solvers/linalg/include
  ${CMAKE_BINARY_DIR}/src/solvers/linalg
)
