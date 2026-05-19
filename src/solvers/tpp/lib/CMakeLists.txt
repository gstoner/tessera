
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
)
