cmake_minimum_required(VERSION 3.29)
project(Trickery CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Trickery main.cu)

target_include_directories(Trickery PRIVATE includes)

set_target_properties(Trickery PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
