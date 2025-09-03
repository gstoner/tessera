
set(LLVM_LINK_COMPONENTS Support)

add_library(TesseraPasses
  CanonicalizeTesseraIR.cpp
  VerifyTesseraIR.cpp
  MigrateTesseraIR.cpp
  Passes.cpp
)

target_include_directories(TesseraPasses
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ir
)

target_link_libraries(TesseraPasses
  PUBLIC
    TesseraIR
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    MLIRTransforms
)

mlir_check_all_link_libraries(TesseraPasses)
