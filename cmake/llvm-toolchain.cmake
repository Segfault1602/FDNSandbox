include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER "/Users/alex/llvm_install/bin/clang++")
set(CMAKE_C_COMPILER "/Users/alex/llvm_install/bin/clang")

set(FDNSANDBOX_SANITIZER $<$<CONFIG:Debug>:-fsanitize=address>)

set(FDNSANDBOX_CXX_COMPILE_OPTIONS
    -Wall -Wextra -Wpedantic -Werror -Wno-sign-compare
    # -Wunsafe-buffer-usage
    ${FDNSANDBOX_SANITIZER})

set(FDNSANDBOX_LINK_OPTIONS ${FDNSANDBOX_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
