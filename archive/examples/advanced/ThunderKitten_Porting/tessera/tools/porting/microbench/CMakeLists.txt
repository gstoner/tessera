
add_executable(tessera_microbench microbench_main.cu)
target_include_directories(tessera_microbench PRIVATE ${CMAKE_SOURCE_DIR}/tessera/include)
set_target_properties(tessera_microbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build experimental WGMMA kernel as a separate object for dev
add_library(tessera_wgmma_exp OBJECT wgmma_experimental.cu)

# (v11.1) No changes needed here; microbench already built.
