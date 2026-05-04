
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  TesseraIR
  TesseraPasses
  tessera_solvers_passes
  tessera_linalg_solver_passes
  TesseraNeighbors
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
  ${CMAKE_SOURCE_DIR}/src/compiler/tessera_neighbors/include
)

# Phase 8: Apple Silicon backend (TESSERA_BUILD_APPLE_BACKEND).
# Only linked when the backend subdirectory has been added by src/CMakeLists.txt.
if(TARGET TesseraApple)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_APPLE_BACKEND)
  target_link_libraries(tessera-opt PRIVATE TesseraApple)
endif()
