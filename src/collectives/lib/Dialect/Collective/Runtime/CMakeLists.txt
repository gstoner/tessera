add_library(TesseraCollectiveRuntime
  Execution.cpp
)
target_include_directories(TesseraCollectiveRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ${CMAKE_CURRENT_BINARY_DIR}/../../..)
target_link_libraries(TesseraCollectiveRuntime PUBLIC ${CMAKE_DL_LIBS})

if(TESSERA_ENABLE_NCCL)
  find_package(CUDAToolkit REQUIRED)
  find_path(TESSERA_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(TESSERA_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(TesseraCollectiveRuntime PUBLIC TESSERA_HAS_NCCL=1)
  target_include_directories(TesseraCollectiveRuntime PUBLIC
    ${TESSERA_NCCL_INCLUDE_DIR})
  target_link_libraries(TesseraCollectiveRuntime PUBLIC
    ${TESSERA_NCCL_LIBRARY} CUDA::cudart)
  if(EXISTS "${TESSERA_NCCL_INCLUDE_DIR}/nccl_device/core.h")
    target_compile_definitions(TesseraCollectiveRuntime PUBLIC
      TESSERA_NCCL_DEVICE_CORE_HEADER=1)
  elseif(EXISTS "${TESSERA_NCCL_INCLUDE_DIR}/nccl_device/core_tmp.h")
    target_compile_definitions(TesseraCollectiveRuntime PUBLIC
      TESSERA_NCCL_DEVICE_CORE_TMP_HEADER=1)
  endif()
endif()

if(TESSERA_ENABLE_RCCL)
  find_path(TESSERA_RCCL_INCLUDE_DIR rccl/rccl.h
    HINTS /opt/rocm/include REQUIRED)
  find_library(TESSERA_RCCL_LIBRARY NAMES rccl
    HINTS /opt/rocm/lib REQUIRED)
  find_library(TESSERA_HIP_RUNTIME_LIBRARY NAMES amdhip64
    HINTS /opt/rocm/lib REQUIRED)
  target_compile_definitions(TesseraCollectiveRuntime PUBLIC
    TESSERA_HAS_RCCL=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(TesseraCollectiveRuntime PUBLIC
    ${TESSERA_RCCL_INCLUDE_DIR})
  target_link_libraries(TesseraCollectiveRuntime PUBLIC
    ${TESSERA_RCCL_LIBRARY} ${TESSERA_HIP_RUNTIME_LIBRARY})
  if(EXISTS "${TESSERA_RCCL_INCLUDE_DIR}/nccl_device/core.h")
    target_compile_definitions(TesseraCollectiveRuntime PUBLIC
      TESSERA_NCCL_DEVICE_CORE_HEADER=1)
  elseif(EXISTS "${TESSERA_RCCL_INCLUDE_DIR}/nccl_device/core_tmp.h")
    target_compile_definitions(TesseraCollectiveRuntime PUBLIC
      TESSERA_NCCL_DEVICE_CORE_TMP_HEADER=1)
  endif()
endif()
