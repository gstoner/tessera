
add_subdirectory(Dialect)
add_subdirectory(Passes)
add_library(TesseraTPPInit STATIC InitTPP.cpp)
target_link_libraries(TesseraTPPInit PUBLIC
  MLIRIR
  MLIRPass
  TesseraTPPDialect
  TesseraTPPPasses
)
target_include_directories(TesseraTPPInit PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/include/tpp/Dialect/TPP
)

# Glass-jaw #2 (2026-06-01) — in-process C ABI shared library so the
# Python TPP driver runs the `tpp-space-time` pipeline via ctypes
# instead of subprocessing `tessera-opt`. Mirrors the Apple GPU runtime
# ctypes pattern; `tessera.solvers.tpp.solve()` loads it from the build
# tree.
add_library(tessera_tpp_capi SHARED TPPCApi.cpp)
target_link_libraries(tessera_tpp_capi PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRLinalgDialect
  TesseraTPPInit
  TesseraTPPDialect
  TesseraTPPPasses
)
target_include_directories(tessera_tpp_capi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/include/tpp/Dialect/TPP
)
