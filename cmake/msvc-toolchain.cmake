include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER cl)
set(CMAKE_C_COMPILER cl)

set(FDNSANDBOX_COMPILE_DEFINITION $<$<CONFIG:Debug>:-D_MSVC_STL_HARDENING=1>
                                  $<$<CONFIG:RelWithDebInfo>:-D_MSVC_STL_HARDENING=1>
)

set(FDNSANDBOX_CXX_COMPILE_OPTIONS
    /WX
    /wd4018
    /wd4244
    /wd4068
    /wd4267
    /wd5030
    /wd4305
    /wd4820
    /wd4514
    /wd5027
    /wd4626
    /wd5026
    /wd4625
    /wd4365
    /wd4710
    /wd5045
    /wd4866
    /GS-
    # /fp:fast
    /Gw
    /Gy
    /GL
    /favor:INTEL64
)

set(FDNSANDBOX_LINK_OPTIONS /INCREMENTAL:NO /LTCG)

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
