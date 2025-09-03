
# Tessera IR Passes — /src layout (v0.2)

Drop this into the repo root. It places the passes beneath `/src` and tests beneath `/tests`.

Build:
  mkdir -p build && cd build
  cmake -G Ninja -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir ..
  ninja

Test:
  export PATH="$PWD/bin:$PATH"
  llvm-lit ../tests/tessera-ir -v
