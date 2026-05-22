
add_executable(tessera-cpx-opt tessera-cpx-opt.cpp)
target_link_libraries(tessera-cpx-opt PRIVATE TesseraCPX MLIRIR MLIRParser MLIRPass MLIRSupport)
install(TARGETS tessera-cpx-opt)
