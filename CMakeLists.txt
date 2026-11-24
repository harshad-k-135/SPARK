cmake_minimum_required(VERSION 3.24)
project(SPARK VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target CUDA compute capabilities")

add_library(spark_build_options INTERFACE)
if(MSVC)
    target_compile_options(spark_build_options INTERFACE $<$<CONFIG:Release>:/O2> /utf-8)
else()
    target_compile_options(spark_build_options INTERFACE $<$<CONFIG:Release>:-O3>)
endif()
include(CTest)

add_library(spark_tokenizer engine/tokenizer/tokenizer.cpp)
target_include_directories(spark_tokenizer PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(spark_tokenizer PUBLIC spark_build_options)
