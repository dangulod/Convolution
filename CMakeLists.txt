cmake_minimum_required(VERSION 2.8)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Ore LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    "main.cpp"
    poipar.cu)

include_directories("/usr/local/cuda/include")
