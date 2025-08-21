add_library(tessera_runtime_cpu OBJECT
  src/api_cpu.cpp
)
target_include_directories(tessera_runtime_cpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TESSERA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(tessera_runtime_cuda OBJECT
    src/cuda/kernels.cu
    src/cuda/gemm_wmma.cu
    src/cuda/gemm_wgmma_sm90.cu
    src/api_cuda.cpp
  )
  target_include_directories(tessera_runtime_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tessera_runtime_cuda PUBLIC CUDA::cudart)
  set_target_properties(tessera_runtime_cuda PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
  )
endif()

add_library(tessera_runtime SHARED
  $<TARGET_OBJECTS:tessera_runtime_cpu>
  $<$<BOOL:${TESSERA_ENABLE_CUDA}>:$<TARGET_OBJECTS:tessera_runtime_cuda>>
)
target_include_directories(tessera_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tessera_runtime PUBLIC
  $<$<BOOL:${TESSERA_ENABLE_CUDA}>:CUDA::cudart>
)
