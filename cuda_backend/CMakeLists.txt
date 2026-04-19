cmake_minimum_required(VERSION 3.18)
project(gpuaudio_features LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_library(gpuaudio_features SHARED audio_features.cu)
target_include_directories(gpuaudio_features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpuaudio_features PRIVATE CUDA::cudart CUDA::cufft)

target_compile_options(gpuaudio_features PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
