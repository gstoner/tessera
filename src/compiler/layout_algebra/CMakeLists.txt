add_library(tessera_layout_algebra SHARED
  LayoutAlgebra.cpp
)

target_include_directories(tessera_layout_algebra
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(tessera_layout_algebra PROPERTIES
  OUTPUT_NAME tessera_layout_algebra
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

if(TESSERA_BUILD_TESTS)
  add_executable(tessera-layout-algebra-test LayoutAlgebraTest.cpp)
  target_link_libraries(tessera-layout-algebra-test PRIVATE tessera_layout_algebra)
endif()

install(TARGETS tessera_layout_algebra
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
