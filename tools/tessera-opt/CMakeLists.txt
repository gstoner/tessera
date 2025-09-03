
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  TesseraIR
  TesseraPasses
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIROptLib
)
