
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRLinalgDialect
  MLIROptLib
)

if(NOT (TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA))
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_CORE_TESSERA_IR)
  target_link_libraries(tessera-opt PRIVATE
    TesseraIR
    TesseraPasses
  )
endif()

if(TARGET tessera_solvers_passes AND TARGET tessera_linalg_solver_passes)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_SOLVERS)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/core/passes
    ${CMAKE_SOURCE_DIR}/src/solvers/linalg/include
    ${CMAKE_BINARY_DIR}/src/solvers/linalg
  )
  target_link_libraries(tessera-opt PRIVATE
    tessera_solvers_passes
    tessera_linalg_solver_passes
  )
endif()

if(TARGET TesseraNeighbors)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_NEIGHBORS)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tessera_neighbors/include
  )
  target_link_libraries(tessera-opt PRIVATE TesseraNeighbors)
endif()

# Phase 8: Apple Silicon backend (TESSERA_BUILD_APPLE_BACKEND).
# Only linked when the backend subdirectory has been added by src/CMakeLists.txt.
if(TARGET TesseraApple)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_APPLE_BACKEND)
  target_link_libraries(tessera-opt PRIVATE TesseraApple)
endif()

# ROCm backend: hardware-free Target IR / ROCDL artifact pipeline.
if(TARGET TesseraROCMConversion)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_ROCM_BACKEND)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/include
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/lib/IR
  )
  target_link_libraries(tessera-opt PRIVATE TesseraROCMIR TesseraROCMConversion)
endif()
