#!/usr/bin/env bash
# NVIDIA-TEST-7: local exact-device proof gate for the RTX 5070 Ti (sm_120).
#
# Keeps CPU, compiler-artifact, device-correctness, and serial-performance
# evidence separate. Reports are retained locally; this project intentionally
# does not execute private GPU proofs through GitHub-hosted/self-hosted runners.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAYER=all
if [ "${1:-}" = "--layer" ]; then
  LAYER="${2:-}"
  shift 2
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--layer cpu|compiler|device|performance|all]" >&2
  exit 2
fi
case "$LAYER" in
  cpu|compiler|device|performance|all) ;;
  *) echo "unknown NVIDIA proof layer: $LAYER" >&2; exit 2 ;;
esac

layer_enabled() {
  [ "$LAYER" = all ] || [ "$LAYER" = "$1" ]
}

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
RUN_ID="${TESSERA_NVIDIA_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)}"
REPORT_DIR="${TESSERA_NVIDIA_REPORT_DIR:-$ROOT/artifacts/nvidia-release/$RUN_ID/$LAYER}"
LOCK_FILE="${TESSERA_NVIDIA_LOCK_FILE:-/tmp/tessera-nvidia-sm120-release.lock}"
LLVM_ROOT="${LLVM_ROOT:-/usr/lib/llvm-23}"
LLVM_DIR="${LLVM_DIR:-$LLVM_ROOT/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-$LLVM_ROOT/lib/cmake/mlir}"
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
command -v flock >/dev/null 2>&1 || {
  echo "NVIDIA release gate requires flock for host concurrency control" >&2
  exit 2
}
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another NVIDIA exact-device proof owns $LOCK_FILE" >&2
  exit 3
fi
status_record() {
  rc=$1
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    status=success
  else
    status=failed
  fi
  printf 'status=%s\nlayer=%s\ncommit=%s\nexit_code=%s\n' \
    "$status" "$LAYER" "$(git rev-parse HEAD)" "$rc" >"$REPORT_DIR/status.txt"
  exit "$rc"
}
trap 'status_record $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
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
if [ -z "$LLVM_LIT" ] && [ -x "$LLVM_ROOT/bin/lit" ]; then
  LLVM_LIT="$LLVM_ROOT/bin/lit"
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
Layer: $LAYER

This directory is one local WSL exact-device proof bundle. Attach or summarize
it in the coordinating PR; it is not GitHub-hosted or self-hosted-runner
evidence.
EOF

if [ "$LAYER" != cpu ]; then
  PATH="$CUDA_BIN:$PATH" cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
    -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
    -DCMAKE_CUDA_COMPILER="$CUDA_BIN/nvcc" \
    -DCUDAToolkit_ROOT="$(dirname "$CUDA_BIN")" \
    -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
    -DTESSERA_ENABLE_CUDA=ON \
    -DTESSERA_CUDA_ARCH=sm_120a \
    -DTESSERA_BUILD_EXAMPLES=OFF \
    -DLLVM_DIR="$LLVM_DIR" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_LIT"
  PATH="$CUDA_BIN:$PATH" ninja -C "$BUILD_DIR" tessera-opt tessera-nvidia-opt \
    tessera_nvidia_gemm tessera_runtime check-tessera-nvidia
fi

export PATH="$CUDA_BIN:$PATH"
export TESSERA_OPT="$BUILD_DIR/tools/tessera-opt/tessera-opt"
export TESSERA_NVIDIA_OPT="$BUILD_DIR/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt"
export MLIR_OPT="${MLIR_OPT:-$LLVM_ROOT/bin/mlir-opt}"
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if layer_enabled cpu; then
  # Ordinary PR CI owns the full cross-backend CPU suite. This physical-host
  # release layer owns NVIDIA host-free contracts plus the shared registry and
  # audit gates that can invalidate an NVIDIA promotion.
  TESSERA_OPT="$ROOT/.nvidia-cpu-no-compiler" \
  TESSERA_NVIDIA_OPT="$ROOT/.nvidia-cpu-no-compiler" \
  MLIR_OPT="$ROOT/.nvidia-cpu-no-compiler" \
    "$PYTHON" -m pytest tests/unit/test_nvidia*.py \
      tests/unit/test_canonical_dtype.py \
      tests/unit/test_tensor_attributes_dtype_audit.py \
      tests/unit/test_operator_registry_foundation.py \
      tests/unit/test_diagnostic_code_registry.py \
      tests/unit/test_pass_metadata.py tests/unit/test_audit_docs.py \
      tests/unit/test_test_suite_architecture.py \
      -m "not slow and not performance and not hardware_nvidia" \
      -q -n 0 --timeout=180 --junitxml="$REPORT_DIR/cpu.xml" || exit $?
fi
if layer_enabled compiler; then
  "$PYTHON" -m pytest tests/unit tests/device/nvidia \
    -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
    --junitxml="$REPORT_DIR/compiler-artifact.xml" || exit $?
fi
if layer_enabled device; then
  "$PYTHON" -m pytest tests/unit tests/device/nvidia tests/integration \
    -m "hardware_nvidia and not performance" -q --durations=100 \
    --junitxml="$REPORT_DIR/device-correctness-1.xml" || exit $?
  "$PYTHON" -m pytest tests/unit tests/device/nvidia tests/integration \
    -m "hardware_nvidia and not performance" -q --durations=100 \
    --junitxml="$REPORT_DIR/device-correctness-2.xml" || exit $?
fi
if layer_enabled performance; then
  # Never add xdist here: performance timing must be serial and isolated.
  "$PYTHON" -m pytest tests/unit tests/device/nvidia tests/performance/nvidia \
    tests/integration -m "hardware_nvidia and performance" \
    -q -n 0 --durations=0 --junitxml="$REPORT_DIR/performance.xml" || exit $?
  mkdir -p "$REPORT_DIR/baselines"
  cp benchmarks/baselines/nvidia_sm120*.json "$REPORT_DIR/baselines/"
  cp benchmarks/baselines/autotune_corpus.json "$REPORT_DIR/baselines/"
  sha256sum "$REPORT_DIR"/baselines/*.json >"$REPORT_DIR/baseline-sha256.txt"
fi
