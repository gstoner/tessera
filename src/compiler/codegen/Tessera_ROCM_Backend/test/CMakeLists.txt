set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Locate a lit runner through the shared validator (cmake/TesseraLit.cmake):
# a candidate must actually run `--version`.  It searches the project venv too:
# `lit` is commonly installed ONLY there, and a non-interactive configure does
# not activate it.  This suite is the ROCm backend's ONLY automated fixture
# coverage (CI does not run it), so "no working runner" makes the check target
# FAIL with the rejected candidates instead of echoing "skipping" and exiting 0.
include(${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/TesseraLit.cmake)
tessera_lit_command(_tessera_rocm_lit)

# Locate the LLVM tools dir (FileCheck / not / count) the fixtures invoke; the
# lit config prepends it to the RUN-line PATH.  Prefer the find_package value,
# fall back to deriving it from FileCheck's location.
if(NOT TESSERA_LLVM_TOOLS_DIR)
  if(LLVM_TOOLS_BINARY_DIR)
    set(TESSERA_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
  else()
    find_program(TESSERA_FILECHECK NAMES FileCheck)
    if(TESSERA_FILECHECK)
      get_filename_component(TESSERA_LLVM_TOOLS_DIR "${TESSERA_FILECHECK}" DIRECTORY)
    endif()
  endif()
endif()

set(TESSERA_ROCM_OPT_EXECUTABLE
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
if(_tessera_rocm_lit)
  add_custom_target(check-tessera-rocm
    COMMAND ${_tessera_rocm_lit} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tessera-rocm-opt)
else()
  tessera_add_lit_unavailable_target(check-tessera-rocm "the ROCm backend lit suite")
  add_dependencies(check-tessera-rocm tessera-rocm-opt)
endif()
