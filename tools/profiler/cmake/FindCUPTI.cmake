# tools/profiler/cmake/FindCUPTI.cmake
find_path(CUPTI_INCLUDE_DIR cupti.h HINTS ENV CUDA_PATH ENV CUDA_HOME PATH_SUFFIXES extras/CUPTI/include include)
find_library(CUPTI_LIBRARY NAMES cupti HINTS ENV CUDA_PATH ENV CUDA_HOME PATH_SUFFIXES extras/CUPTI/lib64 lib64 lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUPTI DEFAULT_MSG CUPTI_INCLUDE_DIR CUPTI_LIBRARY)
if (CUPTI_FOUND)
  add_library(CUPTI::cupti UNKNOWN IMPORTED)
  set_target_properties(CUPTI::cupti PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUPTI_INCLUDE_DIR}"
    IMPORTED_LOCATION "${CUPTI_LIBRARY}")
endif()
