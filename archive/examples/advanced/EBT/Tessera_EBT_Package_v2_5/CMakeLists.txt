cmake_minimum_required(VERSION 3.20)
project(tessera_ebt_v25 LANGUAGES CXX)
add_executable(run_ebt_smoke tools/run_ebt_smoke.cpp)
target_compile_features(run_ebt_smoke PRIVATE cxx_std_17)
