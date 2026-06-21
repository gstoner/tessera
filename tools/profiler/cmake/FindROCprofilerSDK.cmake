find_path(ROCPROFILER_SDK_INCLUDE_DIR
  rocprofiler-sdk/rocprofiler.h
  HINTS ENV ROCM_PATH ENV ROCM_HOME
  PATH_SUFFIXES include)

find_library(ROCPROFILER_SDK_LIBRARY
  NAMES rocprofiler-sdk rocprofiler_sdk
  HINTS ENV ROCM_PATH ENV ROCM_HOME
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCprofilerSDK
  REQUIRED_VARS ROCPROFILER_SDK_INCLUDE_DIR ROCPROFILER_SDK_LIBRARY)

if (ROCprofilerSDK_FOUND AND NOT TARGET ROCprofilerSDK::rocprofiler-sdk)
  add_library(ROCprofilerSDK::rocprofiler-sdk UNKNOWN IMPORTED)
  set_target_properties(ROCprofilerSDK::rocprofiler-sdk PROPERTIES
    IMPORTED_LOCATION "${ROCPROFILER_SDK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ROCPROFILER_SDK_INCLUDE_DIR}")
endif()
