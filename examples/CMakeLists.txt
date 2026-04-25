# Tessera examples
# Driven by the top-level TESSERA_BUILD_EXAMPLES option.
#
# Only subdirectories with a CMakeLists.txt are included.
# Markdown/Python-only examples are self-contained and require no build step.

# ------------------------------------------------------------------
# Optimization tutorial (C++ / CUDA / MLIR code examples)
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/optimization/CMakeLists.txt)
    add_subdirectory(optimization)
endif()

# ------------------------------------------------------------------
# Advanced examples with buildable C++ content
# ------------------------------------------------------------------
add_subdirectory(advanced)
