# TPP space-time numerical correctness sentinel.  Self-contained (standard
# library only, no MLIR/LLVM deps) so it builds and runs without the full
# toolchain — the CPU correctness gate for the stencil / BC / time-step
# semantics the TPP passes denote.
add_executable(tpp-correctness correctness_microbench.cpp)
set_target_properties(tpp-correctness PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
