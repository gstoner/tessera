add_custom_target(check-p3d COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)
add_subdirectory(FileCheck)
