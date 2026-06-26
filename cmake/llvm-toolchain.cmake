include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@22/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@22/bin/clang")
else()

endif()

if(WIN32)
    set(CMAKE_SYSTEM_LIBRARY_PATH "$ENV{ProgramFiles}/LLVM/lib")
endif()
