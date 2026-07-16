if(NOT LLVM_EXTERNAL_LIT)
  find_program(TESSERA_COLLECTIVES_LIT NAMES llvm-lit lit.py lit)
  if(TESSERA_COLLECTIVES_LIT)
    set(LLVM_EXTERNAL_LIT "${TESSERA_COLLECTIVES_LIT}")
  endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")
  add_lit_testsuite(check-tessera-collective "Run Tessera collective tests"
    ${CMAKE_CURRENT_SOURCE_DIR})
else()
  add_custom_target(check-tessera-collective
    COMMAND ${CMAKE_COMMAND} -E echo
            "No Tessera collective lit suite is defined - runtime smoke is the v1.4 validation lane")
endif()
