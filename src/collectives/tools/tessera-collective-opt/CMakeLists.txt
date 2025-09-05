add_llvm_executable(tessera-collective-opt tessera-collective-opt.cpp)
llvm_update_compile_flags(tessera-collective-opt)
target_link_libraries(tessera-collective-opt PRIVATE MLIRIR MLIRParser MLIRPass MLIRSupport MLIRTransforms)
