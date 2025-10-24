include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (APPLE)
        set(FDNSANDBOX_SANITIZER -fsanitize=address)
    endif()
endif()

if (WIN32)
    set(CMAKE_SYSTEM_LIBRARY_PATH "$ENV{ProgramFiles}/LLVM/lib")
endif()

set(FDNSANDBOX_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    -Wno-language-extension-token
    # -Wunsafe-buffer-usage
    ${FDNSANDBOX_SANITIZER}
)

set(FDNSANDBOX_LINK_OPTIONS ${FDNSANDBOX_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
