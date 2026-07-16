#!/usr/bin/env bash
# NVIDIA-TEST-7: local exact-device proof gate for the RTX 5070 Ti (sm_120).
#
# Keeps CPU, compiler-artifact, device-correctness, and serial-performance
# evidence separate.  All reports live under $TESSERA_NVIDIA_REPORT_DIR so a
# CI workflow can retain one complete proof bundle for each run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
LOCAL_VENV="${TESSERA_NVIDIA_VENV:-$ROOT/.venv}"
if [ -z "$PYTHON" ]; then
  if [ -x "$LOCAL_VENV/bin/python" ]; then
    PYTHON="$LOCAL_VENV/bin/python"
  else
    PYTHON=python3
  fi
fi
BUILD_DIR="${TESSERA_NVIDIA_BUILD_DIR:-$ROOT/build-nvidia-cuda}"
REPORT_DIR="${TESSERA_NVIDIA_REPORT_DIR:-$ROOT/artifacts/nvidia-release}"
MLIR_DIR="${MLIR_DIR:-/usr/lib/llvm-22/lib/cmake/mlir}"
CUDA_BIN="${CUDA_BIN:-}"
LLVM_LIT="${LLVM_LIT:-}"

# A systemd user service has a deliberately minimal PATH.  Prefer an explicit
# override, then the normal CUDA symlink, then versioned toolkit installs.
if [ -z "$CUDA_BIN" ]; then
  CUDA_BIN="$(dirname "$(command -v nvcc || true)")"
fi
if [ -z "$CUDA_BIN" ] || [ ! -x "$CUDA_BIN/nvcc" ]; then
  for candidate in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -x "$candidate/nvcc" ] && [ -x "$candidate/ptxas" ]; then
      CUDA_BIN="$candidate"
      break
    fi
  done
fi

mkdir -p "$REPORT_DIR"
if ! "$PYTHON" -c 'import pytest, xdist, hypothesis' >/dev/null 2>&1; then
  # Exact-device proof is a local pre-PR operation.  Its venv belongs to this
  # checkout; no GitHub Actions runner state or service is involved.
  PYTHON=python3 bash scripts/install_test_deps.sh --venv
  PYTHON="$LOCAL_VENV/bin/python"
fi
if [ ! -x "$CUDA_BIN/nvcc" ] || [ ! -x "$CUDA_BIN/ptxas" ]; then
  echo "NVIDIA release gate requires nvcc and ptxas; set CUDA_BIN to the toolkit bin directory" >&2
  exit 2
fi
export PATH="$CUDA_BIN:/usr/lib/wsl/lib:$PATH"
if [ ! -f "$MLIR_DIR/MLIRConfig.cmake" ]; then
  echo "NVIDIA release gate requires MLIR_DIR/MLIRConfig.cmake; got $MLIR_DIR" >&2
  exit 2
fi
if [ -z "$LLVM_LIT" ]; then
  LLVM_LIT="$(command -v llvm-lit || command -v lit || true)"
fi
if [ -z "$LLVM_LIT" ] && [ -x /usr/lib/llvm-22/bin/lit ]; then
  LLVM_LIT=/usr/lib/llvm-22/bin/lit
fi
if [ -z "$LLVM_LIT" ] || [ ! -x "$LLVM_LIT" ]; then
  echo "NVIDIA release gate requires LLVM lit; set LLVM_LIT or install lit on the NVIDIA runner" >&2
  exit 2
fi

{
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,clocks.current.graphics,clocks.current.memory,pstate,power.draw,power.limit --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  "$CUDA_BIN/nvcc" --version
  "$CUDA_BIN/ptxas" --version
  "$PYTHON" --version
  git rev-parse HEAD
} >"$REPORT_DIR/machine-identity.txt"
cat >"$REPORT_DIR/README.md" <<EOF
# NVIDIA exact-device proof bundle

Commit: $(git rev-parse HEAD)
Target: RTX 5070 Ti / sm_120

This directory is the local pre-PR evidence bundle.  Attach or summarize it in
the PR after all four JUnit layers complete; it is not a GitHub Actions artifact.
EOF

PATH="$CUDA_BIN:$PATH" cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_120a \
  -DTESSERA_BUILD_EXAMPLES=OFF \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_EXTERNAL_LIT="$LLVM_LIT"
PATH="$CUDA_BIN:$PATH" ninja -C "$BUILD_DIR" tessera-opt tessera-nvidia-opt \
  tessera_nvidia_gemm tessera_runtime check-tessera-nvidia

export PATH="$CUDA_BIN:$PATH"
export TESSERA_OPT="$BUILD_DIR/tools/tessera-opt/tessera-opt"
export TESSERA_NVIDIA_OPT="$BUILD_DIR/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt"
export MLIR_OPT="${MLIR_OPT:-/usr/lib/llvm-22/bin/mlir-opt}"
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# CPU state is deliberately separate from any device proof.
"$PYTHON" scripts/run_unit_tests.py --timeout=180 -q \
  --junitxml="$REPORT_DIR/cpu.xml"
"$PYTHON" -m pytest tests/unit tests/device/nvidia \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml="$REPORT_DIR/compiler-artifact.xml"
"$PYTHON" -m pytest tests/unit tests/device/nvidia tests/integration -m "hardware_nvidia and not performance" \
  -q --durations=100 --junitxml="$REPORT_DIR/device-correctness-1.xml"
"$PYTHON" -m pytest tests/unit tests/device/nvidia tests/integration -m "hardware_nvidia and not performance" \
  -q --durations=100 --junitxml="$REPORT_DIR/device-correctness-2.xml"
# Never add xdist here: performance timing must be serial and isolated.
"$PYTHON" -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration -m "hardware_nvidia and performance" \
  -q -n 0 --durations=0 --junitxml="$REPORT_DIR/performance.xml"
