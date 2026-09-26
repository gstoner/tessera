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

# The compiled languages a target's own sources use, from each source's
# LANGUAGE property or, failing that, its extension. Per-target rather than
# every enabled language: a user -O0 in CMAKE_CUDA_FLAGS says nothing about a
# pure C++ library.
function(_tessera_target_languages target out)
  get_target_property(sources ${target} SOURCES)
  set(langs "")
  foreach(src IN LISTS sources)
    if(src MATCHES "^\\$<")
      continue()  # generator expression; resolved at generate time only
    endif()
    get_source_file_property(lang "${src}" TARGET_DIRECTORY ${target} LANGUAGE)
    if(NOT lang OR lang STREQUAL "NOTFOUND")
      get_filename_component(ext "${src}" LAST_EXT)
      string(TOLOWER "${ext}" ext)
      if(ext STREQUAL ".c")
        set(lang C)
      elseif(ext MATCHES "^\\.(cc|cpp|cxx|c\\+\\+)$")
        set(lang CXX)
      elseif(ext STREQUAL ".mm")
        set(lang OBJCXX)
      elseif(ext STREQUAL ".cu")
        set(lang CUDA)
      elseif(ext STREQUAL ".hip")
        set(lang HIP)
      else()
        set(lang "")
      endif()
    endif()
    if(lang MATCHES "^(C|CXX|OBJCXX|HIP|CUDA)$")
      list(APPEND langs ${lang})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES langs)
  list(SORT langs)
  set(${out} "${langs}" PARENT_SCOPE)
endfunction()

function(tessera_runtime_library_optimization target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "tessera_runtime_library_optimization: no target '${target}'")
  endif()
  get_property(done TARGET ${target} PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDED)
  if(done)
    message(FATAL_ERROR "tessera_runtime_library_optimization: '${target}' recorded twice")
  endif()
  set_property(TARGET ${target} PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDED TRUE)
  _tessera_target_languages(${target} langs)
  if(CMAKE_CONFIGURATION_TYPES)
    set(level "multi-config generator: per-configuration flags")
  elseif(NOT langs)
    set(level "no compiled sources recognized")
  else()
    # Decide and record per language (CMake keeps CMAKE_<LANG>_FLAGS separate,
    # so a user -O3 in CMAKE_CXX_FLAGS says nothing about CUDA/HIP/OBJCXX).
    set(entries "")
    set(all_default TRUE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" upper)
    foreach(lang IN LISTS langs)
      set(flags "${CMAKE_${lang}_FLAGS}")
      if(CMAKE_BUILD_TYPE)
        set(flags "${flags} ${CMAKE_${lang}_FLAGS_${upper}}")
        list(APPEND entries "${lang}=${CMAKE_BUILD_TYPE}: ${flags}")
        set(all_default FALSE)
      elseif(flags MATCHES "(^| )-O")
        # The user chose this language's level; never override it (an -O3 must
        # not be downgraded to this helper's -O2).
        list(APPEND entries "${lang}=${flags}")
        set(all_default FALSE)
      else()
        if(lang STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
          # -Xcompiler is nvcc's spelling; device code is optimized by nvcc.
          target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
        else()
          target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:${lang}>:-O2>)
        endif()
        list(APPEND entries "${lang}=O2 default")
      endif()
    endforeach()
    if(all_default)
      set(level "O2 (runtime-library default: tree has no build type)")
    else()
      list(JOIN entries " | " joined)
      set(level "languages: ${joined}")
    endif()
  endif()
  # The record is a CMake list joined into JSON, so no list separator, quote
  # or backslash may appear inside it (a backslash is an invalid JSON escape).
  string(REPLACE ";" " " level "${level}")
  string(REPLACE "\\" "/" level "${level}")
  string(REPLACE "\"" "'" level "${level}")
  string(STRIP "${level}" level)
  set_property(GLOBAL APPEND PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDS
    "\"${target}\": \"${level}\"")
endfunction()

# Called once from the top-level CMakeLists after every subdirectory, so the
# file lists every runtime library this tree configured. Written only when its
# content changes: a no-op reconfigure must not make the record newer than a
# library that did not need relinking, or the evidence recorders' freshness
# check (library newer than its record) refuses a current library.
function(tessera_write_runtime_library_build_record)
  get_property(records GLOBAL PROPERTY TESSERA_RUNTIME_LIBRARY_RECORDS)
  list(JOIN records ",\n    " body)
  set(content "{
  \"schema\": \"tessera.runtime_library_build.v1\",
  \"cmake_build_type\": \"${CMAKE_BUILD_TYPE}\",
  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",
  \"libraries\": {
    ${body}
  }
}
")
  set(path "${CMAKE_BINARY_DIR}/runtime_library_build.json")
  if(EXISTS "${path}")
    file(READ "${path}" existing)
    if(existing STREQUAL content)
      return()
    endif()
  endif()
  file(WRITE "${path}" "${content}")
endfunction()
