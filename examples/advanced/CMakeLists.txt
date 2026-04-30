# Advanced examples — only subdirectories that have a CMakeLists.txt are built.
# Markdown/Python-only examples are self-contained and require no build step.

foreach(_example
    power_retention
)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_example}/CMakeLists.txt)
        add_subdirectory(${_example})
    endif()
endforeach()
