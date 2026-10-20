#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_HOME:?source scripts/_nvidia_env.sh first}"
: "${TESSERA_RELU_RUNTIME_OUTPUT:?set a new output path under scratch}"
test ! -e "$TESSERA_RELU_RUNTIME_OUTPUT"
mkdir -p "$(dirname "$TESSERA_RELU_RUNTIME_OUTPUT")"
g++ -std=c++17 -O3 -DNDEBUG -fPIC -shared -pthread \
  -I "$CUDA_HOME/include" -I src/runtime/include \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp \
  -L/usr/lib/wsl/lib -Wl,-rpath,/usr/lib/wsl/lib -lcuda -ldl \
  -o "$TESSERA_RELU_RUNTIME_OUTPUT"
sha256sum "$TESSERA_RELU_RUNTIME_OUTPUT"
