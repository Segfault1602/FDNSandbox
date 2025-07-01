cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20)

project(FDNToolbox)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)

# if (APPLE)
#     list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
# endif()

include(./fetch_deps.cmake)

add_subdirectory(externals/audio_utils)
add_subdirectory(externals/sffdn)

add_subdirectory(src)

# add_subdirectory(src)