cmake_minimum_required(VERSION 4.1.1)
set(CMAKE_CXX_STANDARD 20)

project(FDNSandbox)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif()

option(FDN_SANDBOX_ENABLE_HARDENING "Enable hardening" ON)
option(FDN_SANDBOX_USE_SANITIZER "Use AddressSanitizer" OFF)

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

include(./fetch_deps.cmake)

add_subdirectory(src)
