#!/usr/bin/env bash
# Tessera build script.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_dir="${BUILD_DIR:-build}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
enable_cuda="${TESSERA_ENABLE_CUDA:-OFF}"
enable_hip="${TESSERA_ENABLE_HIP:-OFF}"
build_tests="${TESSERA_BUILD_TESTS:-ON}"
build_examples="${TESSERA_BUILD_EXAMPLES:-ON}"
build_python="${TESSERA_BUILD_PYTHON:-ON}"

if [[ "${enable_cuda}" == "ON" || "${enable_hip}" == "ON" ]]; then
    cpu_only="${TESSERA_CPU_ONLY:-OFF}"
else
    cpu_only="${TESSERA_CPU_ONLY:-ON}"
fi

if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
else
    jobs="4"
fi

# --- Locate LLVM/MLIR 23 -----------------------------------------------------
# find_package(LLVM CONFIG) needs a hint on most setups. Resolve a prefix from
# (1) an explicit LLVM_DIR/LLVM_PREFIX env, (2) Homebrew (macOS), (3) the
# apt.llvm.org LLVM 23 prefix at /usr/lib/llvm-23 (Ubuntu), or
# (4) llvm-config on PATH.
llvm_dir="${LLVM_DIR:-}"
mlir_dir="${MLIR_DIR:-}"
if [[ -z "${llvm_dir}" ]]; then
    llvm_prefix="${LLVM_PREFIX:-}"
    if [[ -z "${llvm_prefix}" ]] && command -v brew >/dev/null 2>&1; then
        llvm_prefix="$(brew --prefix llvm@23 2>/dev/null || true)"
    fi
    if [[ -z "${llvm_prefix}" ]]; then
        d=/usr/lib/llvm-23
        [[ -d "${d}/lib/cmake/llvm" ]] && llvm_prefix="${d}"
    fi
    if [[ -z "${llvm_prefix}" ]] && command -v llvm-config >/dev/null 2>&1; then
        llvm_major="$(llvm-config --version | cut -d. -f1)"
        [[ "${llvm_major}" == "23" ]] && llvm_prefix="$(llvm-config --prefix)"
    fi
    if [[ -n "${llvm_prefix}" ]]; then
        llvm_dir="${llvm_prefix}/lib/cmake/llvm"
        mlir_dir="${llvm_prefix}/lib/cmake/mlir"
    fi
fi
[[ -z "${mlir_dir}" && -n "${llvm_dir}" ]] && mlir_dir="$(dirname "${llvm_dir}")/mlir"

# ROCm prefix so find_package(hip) resolves when HIP is enabled.
rocm_prefix="${TESSERA_ROCM_PREFIX:-/opt/rocm}"
prefix_path="${CMAKE_PREFIX_PATH:-}"
if [[ "${enable_hip}" == "ON" && -d "${rocm_prefix}" ]]; then
    prefix_path="${prefix_path:+${prefix_path};}${rocm_prefix}"
fi

# Default the ROCm Target IR backend on whenever HIP is enabled.
build_rocm_backend="${TESSERA_BUILD_ROCM_BACKEND:-${enable_hip}}"

extra_cmake_args=()
[[ -n "${llvm_dir}" ]] && extra_cmake_args+=("-DLLVM_DIR=${llvm_dir}")
[[ -n "${mlir_dir}" ]] && extra_cmake_args+=("-DMLIR_DIR=${mlir_dir}")
[[ -n "${prefix_path}" ]] && extra_cmake_args+=("-DCMAKE_PREFIX_PATH=${prefix_path}")
extra_cmake_args+=("-DTESSERA_BUILD_ROCM_BACKEND=${build_rocm_backend}")

echo "Building Tessera"
echo "  build dir     : ${build_dir}"
echo "  build type    : ${build_type}"
echo "  CPU only      : ${cpu_only}"
echo "  CUDA          : ${enable_cuda}"
echo "  HIP           : ${enable_hip}"
echo "  ROCm backend  : ${build_rocm_backend}"
echo "  LLVM_DIR      : ${llvm_dir:-<find_package default>}"
echo "  MLIR_DIR      : ${mlir_dir:-<find_package default>}"
echo "  prefix path   : ${prefix_path:-<none>}"
echo "  tests         : ${build_tests}"
echo "  examples      : ${build_examples}"
echo "  python        : ${build_python}"
echo "  jobs          : ${jobs}"

cmake -S "${repo_root}" -B "${repo_root}/${build_dir}" \
    -DTESSERA_CPU_ONLY="${cpu_only}" \
    -DTESSERA_ENABLE_CUDA="${enable_cuda}" \
    -DTESSERA_ENABLE_HIP="${enable_hip}" \
    -DTESSERA_BUILD_TESTS="${build_tests}" \
    -DTESSERA_BUILD_EXAMPLES="${build_examples}" \
    -DTESSERA_BUILD_PYTHON="${build_python}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    "${extra_cmake_args[@]}"

cmake --build "${repo_root}/${build_dir}" --parallel "${jobs}"

echo "Build completed successfully"
