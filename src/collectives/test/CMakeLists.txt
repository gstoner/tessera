# Shared validated lit runner (cmake/TesseraLit.cmake).
#
# Invoke the validated runner DIRECTLY, as every other migrated suite does.
# Do not route it through LLVM's add_lit_testsuite: AddLLVM's add_lit_target
# builds `${Python3_EXECUTABLE};<LLVM_EXTERNAL_LIT>` (read in the LLVM 23.1.1
# AddLLVM.cmake shipped on Princess-Luna and the Mac), so it launches the lit
# script under the *configure* interpreter and bypasses the shebang.  The
# resolver validates `<lit> --version` through the shebang, so a venv-only lit
# (Princess-Luna) passed the probe and then died with
# `ModuleNotFoundError: No module named 'lit'` under the system python3.
include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/TesseraLit.cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")
  tessera_lit_command(_tessera_collectives_lit)
  if(_tessera_collectives_lit)
    add_custom_target(check-tessera-collective
      COMMAND ${_tessera_collectives_lit} -sv ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Run Tessera collective tests"
      USES_TERMINAL
      VERBATIM)
  else()
    tessera_add_lit_unavailable_target(check-tessera-collective
      "the collectives lit suite")
  endif()
else()
  add_custom_target(check-tessera-collective
    COMMAND ${CMAKE_COMMAND} -E echo
            "No Tessera collective lit suite is defined - runtime smoke is the v1.4 validation lane")
endif()
