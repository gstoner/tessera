set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Locate a lit runner.  A fresh find_program into a dedicated cache var avoids a
# polluted/empty ``LLVM_EXTERNAL_LIT`` cache entry (which an earlier configure
# could have left as an empty STRING, defeating the historic re-search).
if(NOT LLVM_EXTERNAL_LIT)
  # Search the project venv too. `lit` is commonly installed ONLY there (it is
  # in `.venv/bin` on the Strix Halo box, absent from the system path), and a
  # non-interactive configure does not activate it — so the bare find_program
  # silently found nothing and this whole suite reported "skipping" while the
  # build exited 0. That matters more here than elsewhere: CI does not run this
  # suite at all, so it is the ROCm backend's ONLY automated fixture coverage,
  # and a silent skip means it is not covered anywhere.
  find_program(TESSERA_LIT
    NAMES llvm-lit lit.py lit
    HINTS "${CMAKE_SOURCE_DIR}/.venv/bin" "$ENV{VIRTUAL_ENV}/bin"
  )
  if(TESSERA_LIT)
    set(LLVM_EXTERNAL_LIT "${TESSERA_LIT}")
  else()
    message(WARNING
      "check-tessera-rocm will SKIP: no lit runner found. This suite is the "
      "ROCm backend's only automated fixture coverage and CI does not run it. "
      "Install lit, or pass -DTESSERA_LIT=/path/to/lit.")
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
