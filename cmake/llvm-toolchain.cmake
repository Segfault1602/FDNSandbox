include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()

set(FDNSANDBOX_SANITIZER $<$<CONFIG:Debug>:-fsanitize=address>)

set(FDNSANDBOX_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    # -Wunsafe-buffer-usage
    ${FDNSANDBOX_SANITIZER}
)

set(FDNSANDBOX_LINK_OPTIONS ${FDNSANDBOX_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
