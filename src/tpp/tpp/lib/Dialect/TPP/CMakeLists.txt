
add_mlir_dialect_library(TesseraTPPDialect
  TPPDialect.cpp
  TPPOps.cpp
  DEPENDS
    MLIRIR
)
