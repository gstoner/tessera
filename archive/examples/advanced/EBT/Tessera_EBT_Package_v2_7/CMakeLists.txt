cmake_minimum_required(VERSION 3.20)
project(tessera_ebt_v27 LANGUAGES CXX)
add_executable(jvp_equals_directional models/ebt/tests/jvp_equals_directional_cpu.cpp)
set_property(TARGET jvp_equals_directional PROPERTY CXX_STANDARD 17)
