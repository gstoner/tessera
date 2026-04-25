# Advanced examples — only subdirectories that have a CMakeLists.txt are built.
# Markdown/Python-only examples are self-contained and require no build step.

foreach(_example
    power_retention
    Tessera_DiffEntropy_Op
    Tessera_KAN_Port
    Tessera_P3D
    ThunderKitten_Porting
    atlas
)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_example}/CMakeLists.txt)
        add_subdirectory(${_example})
    endif()
endforeach()
