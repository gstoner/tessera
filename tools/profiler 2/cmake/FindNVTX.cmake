# tools/profiler/cmake/FindNVTX.cmake
find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h HINTS ENV NVTX_ROOT PATH_SUFFIXES include)
find_library(NVTX_LIBRARY NAMES nvToolsExt nvtx3 HINTS ENV NVTX_ROOT PATH_SUFFIXES lib lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIR NVTX_LIBRARY)
if (NVTX_FOUND) 
  add_library(NVTX::nvtx3 UNKNOWN IMPORTED)
  set_target_properties(NVTX::nvtx3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVTX_INCLUDE_DIR}"
    IMPORTED_LOCATION "${NVTX_LIBRARY}")
endif()
