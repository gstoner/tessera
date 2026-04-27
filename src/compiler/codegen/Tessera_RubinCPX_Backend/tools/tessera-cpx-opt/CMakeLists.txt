
add_executable(tessera-cpx-opt tessera-cpx-opt.cpp)

target_link_libraries(tessera-cpx-opt
  PRIVATE
    TesseraRubinCPX          # dialect + all CPX passes
    MLIRAllDialects
    MLIROptLib
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

install(TARGETS tessera-cpx-opt DESTINATION bin)
