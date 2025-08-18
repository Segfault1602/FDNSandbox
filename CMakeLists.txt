cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20)

project(FDNSandbox)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(./fetch_deps.cmake)

add_subdirectory(externals/audio_utils)

set(SFFDN_BUILD_TESTS OFF)
add_subdirectory(externals/sffdn)

add_subdirectory(src)
