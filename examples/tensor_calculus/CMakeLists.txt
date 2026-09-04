# Reference-only tutorial: visible as a target, with no build dependency.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_custom_target(tensor_calculus_tutorial
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/python
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tensor_calculus_tutorial.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
