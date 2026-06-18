set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Locate a lit runner.  A fresh find_program into a dedicated cache var avoids a
# polluted/empty ``LLVM_EXTERNAL_LIT`` cache entry (which an earlier configure
# could have left as an empty STRING, defeating the historic re-search).
if(NOT LLVM_EXTERNAL_LIT)
  find_program(TESSERA_LIT NAMES llvm-lit lit.py lit)
  if(TESSERA_LIT)
    set(LLVM_EXTERNAL_LIT "${TESSERA_LIT}")
  endif()
endif()

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
if(LLVM_EXTERNAL_LIT)
  add_custom_target(check-tessera-rocm
    COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tessera-rocm-opt)
else()
  add_custom_target(check-tessera-rocm
    COMMAND ${CMAKE_COMMAND} -E echo "llvm-lit not found - skipping Tessera ROCm lit tests"
    DEPENDS tessera-rocm-opt)
endif()
