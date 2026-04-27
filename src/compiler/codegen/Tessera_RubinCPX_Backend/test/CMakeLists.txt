find_program(LLVM_EXTERNAL_LIT llvm-lit lit.py lit)

if(LLVM_EXTERNAL_LIT)
  add_custom_target(check-tessera-rubincpx
    COMMAND ${CMAKE_COMMAND} -E env
      "PATH=$<TARGET_FILE_DIR:tessera-cpx-opt>:$ENV{PATH}"
      ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS tessera-cpx-opt
  )
  add_test(NAME tessera-rubincpx-tests
    COMMAND ${CMAKE_COMMAND} -E env
      "PATH=$<TARGET_FILE_DIR:tessera-cpx-opt>:$ENV{PATH}"
      ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_SOURCE_DIR}
  )
else()
  message(WARNING "llvm-lit not found; Rubin CPX tests disabled")
endif()
