if(MSVC)
    set(FDN_SANDBOX_WARNINGS_CXX /W3 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(FDN_SANDBOX_WARNINGS_CXX
        -Wall
        -Wextra
        -Wpedantic
        -Wno-sign-compare
        -Wno-language-extension-token
        -Wno-gnu-anonymous-struct
    )
endif()

add_library(fdn_sandbox_warnings INTERFACE)
add_library(fdn_sandbox::fdn_sandbox_warnings ALIAS fdn_sandbox_warnings)
target_compile_options(fdn_sandbox_warnings INTERFACE ${FDN_SANDBOX_WARNINGS_CXX})
