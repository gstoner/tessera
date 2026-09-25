#!/usr/bin/env bash
# Build-type inventory: every build tree under the repo, its cached build type
# and flags, and the optimization flags each runtime library actually compiles
# with (read from `ninja -t commands`, so it reports what runs, not what the
# cache implies). Usage: scripts/inventory_build_types.sh [repo root]
# Evidence: benchmarks/baselines/runtime_lib_opt_20260925/README.md.
repo=${1:-$HOME/programming/tessera}; cd "$repo" || exit 1
command -v ninja >/dev/null || { [ -f ~/.config/tessera/env.sh ] && source ~/.config/tessera/env.sh; }
echo "##### host=$(hostname) repo=$repo head=$(git log -1 --format=%h 2>/dev/null)"
for cache in $(ls -d */CMakeCache.txt 2>/dev/null); do
  t=${cache%/CMakeCache.txt}
  get() { grep -E "^$1:" "$cache" | head -1 | cut -d= -f2-; }
  echo "== tree $t  (mtime $(date -r "$cache" +%F))"
  echo "   CMAKE_BUILD_TYPE='$(get CMAKE_BUILD_TYPE)' generator='$(get CMAKE_GENERATOR)'"
  for v in CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_CUDA_FLAGS CMAKE_HIP_FLAGS CMAKE_OBJCXX_FLAGS; do
    val=$(get $v); [ -n "$val" ] && echo "   $v='$val'"
  done
  [ -f "$t/build.ninja" ] || { echo "   (no build.ninja)"; continue; }
  for lib in libtessera_nvidia_fft libtessera_nvidia_gemm libtessera_nvidia_rng libtessera_spectral_rocm \
             libtessera_runtime libtessera_rocm_runtime libtessera_jit libTesseraAppleRuntime libtessera_x86 \
             libtessera_x86_kernels libtessera_elementwise libtessera_x86_elementwise; do
    for out in $(ninja -C "$t" -t targets all 2>/dev/null | cut -d: -f1 | grep -E "(^|/)$lib[^/]*\.(so|dylib)$" | head -3); do
      flags=$(ninja -C "$t" -t commands "$out" 2>/dev/null | grep -E " -c | -x " | tr ' ' '\n' \
              | grep -E "^-O[0-9sgz]?$|^-Xcompiler=-O|^-g$|^--offload-arch|^-DNDEBUG$" | sort | uniq -c | tr '\n' ' ')
      n=$(ninja -C "$t" -t commands "$out" 2>/dev/null | grep -cE " -c | -x ")
      echo "   $out: $n compile cmds; flags: ${flags:-<none>}"
    done
  done
done
