
set(LLVM_LINK_COMPONENTS
  Support
)

add_library(TesseraPasses
  CanonicalizeTesseraIR.cpp
  VerifyTesseraIR.cpp
  Passes.cpp
)

target_include_directories(TesseraPasses
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(TesseraPasses
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
    MLIRTransforms
)

mlir_check_all_link_libraries(TesseraPasses)
