
add_mlir_dialect_library(TesseraTPPDialect
  TPPDialect.cpp
  TPPOps.cpp
  TPPTypes.cpp
  TPPAttrs.cpp
  DEPENDS
    TPPGen
    TPPTypesGen
    TPPAttrsGen
    TPPOpsGen
  LINK_LIBS PUBLIC MLIRIR
)
