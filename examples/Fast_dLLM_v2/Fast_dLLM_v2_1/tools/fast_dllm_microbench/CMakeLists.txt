cmake_minimum_required(VERSION 3.16)
project(fast_dllm_microbench LANGUAGES CXX)
add_executable(fast_dllm_microbench main.cpp)
target_compile_features(fast_dllm_microbench PRIVATE cxx_std_17)
