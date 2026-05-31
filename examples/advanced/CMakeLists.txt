# Advanced examples — only subdirectories that have a CMakeLists.txt are built.
# Markdown/Python-only examples are self-contained and require no build step.

if(TESSERA_ENABLE_CUDA)
    add_subdirectory(power_retention)
else()
    message(STATUS "Skipping advanced/power_retention because TESSERA_ENABLE_CUDA is OFF")
endif()
