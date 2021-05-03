cmake_minimum_required(VERSION 2.8)

project(Ore LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

add_executable(${PROJECT_NAME}
    "main.cpp"
    poipar.cu)

