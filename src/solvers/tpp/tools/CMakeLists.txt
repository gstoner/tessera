# Standalone opt-style driver for the TPP dialect + space-time passes.
add_executable(tessera-tpp-opt tessera-tpp-opt.cpp)
target_link_libraries(tessera-tpp-opt PRIVATE
  TesseraTPPInit
  TesseraTPPDialect
  TesseraTPPPasses
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
)
target_include_directories(tessera-tpp-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/include/tpp/Dialect/TPP
)
