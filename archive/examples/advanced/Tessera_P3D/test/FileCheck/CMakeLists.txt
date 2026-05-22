file(GLOB TEST_FILES *.mlir)
foreach(t ${TEST_FILES})
  add_test(NAME p3d-mlir-${t} COMMAND llvm-lit ${CMAKE_CURRENT_SOURCE_DIR}/${t})
endforeach()
