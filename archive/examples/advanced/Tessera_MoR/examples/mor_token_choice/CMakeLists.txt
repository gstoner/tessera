# examples/mor_token_choice/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

option(TESSERA_ENABLE_LIT "Enable lit tests for MoR sample" ON)

if(TESSERA_ENABLE_LIT)
  find_program(LIT_EXE NAMES lit lit.py llvm-lit)
  if(NOT LIT_EXE)
    message(WARNING "lit not found — MoR sample tests will be skipped.")
  else()
    add_custom_target(check-mor
      COMMAND "${LIT_EXE}" -v "${CMAKE_CURRENT_LIST_DIR}/tests"
      DEPENDS "${CMAKE_CURRENT_LIST_DIR}/mlir/mor_token_choice.mlir"
      WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
      COMMENT "Running MoR lit tests")
  endif()
endif()
