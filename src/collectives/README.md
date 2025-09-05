# Tessera Collectives Planner v1.2
This build extends v1.0/v1.1 with:
- Attention auto-plan pass (`-tessera-attn-autoplan`)
- QoS throttling markers around chunked collectives
- Full 2D tiling support in `-tessera-plan-collectives`

Build:
  mkdir build && cd build
  cmake -G Ninja -DMLIR_DIR=<mlir-cmake> -DLLVM_DIR=<llvm-cmake> ..
  ninja && ninja check-tessera-collective
