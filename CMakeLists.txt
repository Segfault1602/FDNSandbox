cmake_minimum_required(VERSION 4.1.1)
set(CMAKE_CXX_STANDARD 20)

project(FDNSandbox)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif()

include(./fetch_deps.cmake)

add_subdirectory(externals/audio_utils)

set(SFFDN_BUILD_TESTS OFF)
add_subdirectory(externals/sffdn)

add_subdirectory(src)
