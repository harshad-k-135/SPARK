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

add_library(spark_model engine/model_loader/json.cpp engine/model_loader/model.cpp)
target_link_libraries(spark_model PUBLIC spark_tokenizer)

add_library(spark_cpu engine/transformer/cpu_forward.cpp engine/runtime/session.cpp)
add_library(spark_sampling engine/sampling/sampler.cpp)
target_link_libraries(spark_sampling PUBLIC spark_tokenizer)
target_link_libraries(spark_cpu PUBLIC spark_model spark_sampling)
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spark_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(spark engine/runtime/cli.cpp)
target_link_libraries(spark PRIVATE spark_cpu)
if(MINGW)
    target_link_options(spark PRIVATE -municode)
endif()

if(BUILD_TESTING)
    add_executable(spark_check tests/golden/check_reference.cpp)
    target_link_libraries(spark_check PRIVATE spark_cpu)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME spark_cpu_smoke
                 COMMAND "${Python3_EXECUTABLE}"
                 "${PROJECT_SOURCE_DIR}/tools/reference/generate_smoke.py"
                 --check $<TARGET_FILE:spark_check> --cli $<TARGET_FILE:spark>)
    endif()
    set(SPARK_GATE1_MODEL "" CACHE PATH "Exported pinned Meta Llama 3.2 1B model for Gate 1")
    if(SPARK_GATE1_MODEL)
        add_test(NAME spark_gate_1 COMMAND spark_check --gate1 "${SPARK_GATE1_MODEL}"
                 "${PROJECT_SOURCE_DIR}/tests/golden/reference.json")
    endif()
endif()
