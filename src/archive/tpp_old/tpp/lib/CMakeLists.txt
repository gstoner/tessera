
add_subdirectory(Dialect)
add_subdirectory(Passes)
add_library(TesseraTPPInit STATIC InitTPP.cpp)
target_link_libraries(TesseraTPPInit PUBLIC MLIRIR)
