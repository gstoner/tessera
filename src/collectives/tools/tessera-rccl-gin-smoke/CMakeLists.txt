add_executable(tessera-rccl-gin-smoke gin_smoke.cpp)
target_link_libraries(tessera-rccl-gin-smoke PRIVATE TesseraCollectiveRuntime)
target_compile_definitions(tessera-rccl-gin-smoke PRIVATE
  __HIP_PLATFORM_AMD__=1)

if(TESSERA_ENABLE_RCCL_GIN_DEVICE_TEST)
  add_test(NAME tessera-rccl-gin-smoke COMMAND tessera-rccl-gin-smoke)
  set_tests_properties(tessera-rccl-gin-smoke PROPERTIES
    LABELS "hardware_rocm;gfx1151;rccl_gin;multinode"
    SKIP_RETURN_CODE 77)
endif()
