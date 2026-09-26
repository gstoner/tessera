# RUNTIME-LIB-OPT-1 (owner decision 2026-09-26): optimize the runtime kernel
# libraries when a tree has no build type.
#
# Several fleet trees (the canonical configure commands in CLAUDE.md) leave
# CMAKE_BUILD_TYPE empty, which compiles every translation unit with no -O
# flag. For the compiler that is harmless and even useful -- MLIR/LLVM header
# assertions stay live because NDEBUG is not defined. For the runtime kernel
# libraries it is not: they ARE the measured code. On Princess-Luna the x86
# AVX-512 library ran 2.1-3.3x slower at -O0 than at -O3, while Tajasarus built
# the same library Release, so timing packets from the two Zen 5 boxes were not
# comparable (benchmarks/baselines/runtime_lib_opt_20260925/).
#
# tessera_runtime_library_optimization(<target>) therefore adds -O2 to that
# target only, only when no build type is set, and never defines NDEBUG (the
# runtime directories contain no assert()). CUDA device code is already
# optimized by nvcc, so CUDA gets -O2 for host code only. Each call also records
# the target's effective optimization into
# ${CMAKE_BINARY_DIR}/runtime_library_build.json, which benchmark recorders
# stamp into evidence (Decisions #11/#12): a latency from an -O0 library is not
# comparable to one from an optimized library.

set_property(GLOBAL PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDS "")

function(tessera_runtime_library_optimization target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "tessera_runtime_library_optimization: no target '${target}'")
  endif()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX,HIP>:-O2>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
    set(level "O2 (runtime-library default: tree has no build type)")
  elseif(CMAKE_CONFIGURATION_TYPES)
    set(level "multi-config generator: per-configuration flags")
  else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" upper)
    set(level "${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${upper}}")
  endif()
  # The record is a CMake list joined into JSON, so neither list separators
  # nor quotes may appear inside it.
  string(REPLACE ";" " " level "${level}")
  string(REPLACE "\"" "'" level "${level}")
  set_property(GLOBAL APPEND PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDS
    "\"${target}\": \"${level}\"")
endfunction()

# Called once from the top-level CMakeLists after every subdirectory, so the
# file lists every runtime library this tree configured.
function(tessera_write_runtime_library_build_record)
  get_property(records GLOBAL PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDS)
  list(JOIN records ",\n    " body)
  file(WRITE "${CMAKE_BINARY_DIR}/runtime_library_build.json"
"{
  \"schema\": \"tessera.runtime_library_build.v1\",
  \"cmake_build_type\": \"${CMAKE_BUILD_TYPE}\",
  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",
  \"libraries\": {
    ${body}
  }
}
")
endfunction()
