add_executable(tessera-rccl-lsa-smoke lsa_smoke.hip)
set_source_files_properties(lsa_smoke.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(tessera-rccl-lsa-smoke PRIVATE TesseraCollectiveRuntime)
target_compile_definitions(tessera-rccl-lsa-smoke PRIVATE
  __HIP_PLATFORM_AMD__=1)

if(TESSERA_ENABLE_RCCL_LSA_DEVICE_TEST)
  add_test(NAME tessera-rccl-lsa-smoke COMMAND tessera-rccl-lsa-smoke)
  set_tests_properties(tessera-rccl-lsa-smoke PROPERTIES
    LABELS "hardware_rocm;gfx1151;rccl_lsa"
    SKIP_RETURN_CODE 77)
endif()
