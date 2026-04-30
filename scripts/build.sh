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

echo "Building Tessera"
echo "  build dir     : ${build_dir}"
echo "  build type    : ${build_type}"
echo "  CPU only      : ${cpu_only}"
echo "  CUDA          : ${enable_cuda}"
echo "  HIP           : ${enable_hip}"
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
    -DCMAKE_BUILD_TYPE="${build_type}"

cmake --build "${repo_root}/${build_dir}" --parallel "${jobs}"

echo "Build completed successfully"
