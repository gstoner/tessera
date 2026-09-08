#!/usr/bin/env bash
# Isolated COMPILER-DEVEX-1 toolchain. Never replaces the production LLVM.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 SOURCE_DIR BUILD_DIR (JOBS defaults to 8)" >&2
  exit 2
fi
source_dir=$1
build_dir=$2
revision=e7ce3600b55034ddf819638f395e3c475fad5be2
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 --branch llvmorg-23.1.1 https://github.com/llvm/llvm-project.git "$source_dir"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
cmake -S "$source_dir/llvm" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  '-DLLVM_ENABLE_PROJECTS=mlir;lld' '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU' \
  -DLLVM_INCLUDE_TESTS=ON -DMLIR_INCLUDE_TESTS=ON -DLLVM_PARALLEL_LINK_JOBS=2
cmake --build "$build_dir" --target mlir-opt FileCheck llvm-config lld -j"${JOBS:-8}"
[[ $("$build_dir/bin/llvm-config" --assertion-mode) == ON ]]
"$build_dir/bin/mlir-opt" --version
