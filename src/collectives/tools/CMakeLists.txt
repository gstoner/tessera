add_subdirectory(tessera-collective-opt)
add_subdirectory(tessera-exec-demo)
add_subdirectory(tessera-trace-demo)
add_subdirectory(tessera-collective-runtime-smoke)

if(TESSERA_ENABLE_RCCL AND CMAKE_HIP_COMPILER)
  add_subdirectory(tessera-rccl-lsa-smoke)
  add_subdirectory(tessera-rccl-gin-smoke)
endif()
