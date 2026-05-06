
add_mlir_dialect_library(TesseraTPPDialect
  TPPDialect.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
    TPPGen
    TPPTypesGen
    TPPAttrsGen
    TPPOpsGen
  LINK_LIBS PUBLIC MLIRIR
)

target_include_directories(TesseraTPPDialect PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include
  ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
)

target_include_directories(obj.TesseraTPPDialect PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include
  ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
)
