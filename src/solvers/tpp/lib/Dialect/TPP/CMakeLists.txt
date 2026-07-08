
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

# Generated *.inc land under the tablegen `include/` dir; its prefix differs
# between the monorepo (build/src/solvers/tpp/include/...) and a standalone
# build (build/include/...), so both candidate roots are listed.
target_include_directories(TesseraTPPDialect PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include
  ${CMAKE_BINARY_DIR}/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
)

target_include_directories(obj.TesseraTPPDialect PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/src/solvers/tpp/include
  ${CMAKE_BINARY_DIR}/include/tpp/Dialect/TPP
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
)
