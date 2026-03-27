# Dependency that should already be installed on the system:
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP)
find_package(OpenGL REQUIRED)

include(FetchContent)

# Below is the list of third-party dependencies fetched via CPM.cmake

FetchContent_Declare(CPM GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake GIT_TAG v0.42.1)
FetchContent_MakeAvailable(CPM)
include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)

cpmaddpackage(
    NAME
    Boost
    URL
    https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
    OPTIONS
    "BOOST_ENABLE_CMAKE ON"
    "BOOST_SKIP_INSTALL_RULES ON"
    "BUILD_SHARED_LIBS OFF"
    "BOOST_INCLUDE_LIBRARIES dll\\\;math"
)

cpmaddpackage(
    NAME
    GLFW
    GITHUB_REPOSITORY
    glfw/glfw
    GIT_TAG
    3.4
    OPTIONS
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BUILD_DOCS OFF"
)

# find_package(Boost REQUIRED COMPONENTS math dll)
cpmaddpackage(
    NAME
    Eigen
    GIT_TAG
    5.0.1
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen
)

if(Eigen_ADDED)
    get_target_property(_eigen_inc eigen INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(eigen SYSTEM INTERFACE ${_eigen_inc})
endif()

cpmaddpackage(
    NAME
    libsndfile
    GIT_TAG
    master
    GIT_REPOSITORY
    "https://github.com/libsndfile/libsndfile"
    OPTIONS
    "BUILD_PROGRAMS OFF"
    "BUILD_EXAMPLES OFF"
    "BUILD_REGTEST OFF"
    "ENABLE_EXTERNAL_LIBS OFF"
    "BUILD_TESTING OFF"
    "ENABLE_MPEG OFF"
    "ENABLE_CPACK OFF"
    "ENABLE_PACKAGE_CONFIG OFF"
    "INSTALL_PKGCONFIG_MODULE OFF"
)

cpmaddpackage("gh:nlohmann/json@3.12.0")

cpmaddpackage("gh:odygrd/quill@11.0.2")

cpmaddpackage(
    NAME
    imgui
    GIT_TAG
    docking
    GITHUB_REPOSITORY
    ocornut/imgui
    DOWNLOAD_ONLY
    TRUE
)

cpmaddpackage(
    NAME
    implot
    GIT_TAG
    master
    GITHUB_REPOSITORY
    epezent/implot
    DOWNLOAD_ONLY
    TRUE
)

cpmaddpackage(
    NAME
    implot3d
    GIT_TAG
    main
    GITHUB_REPOSITORY
    brenocq/implot3d
    DOWNLOAD_ONLY
    TRUE
)

cpmaddpackage(
    NAME
    imgui-filebrowser
    GIT_TAG
    master
    GITHUB_REPOSITORY
    AirGuanZ/imgui-filebrowser
    DOWNLOAD_ONLY
    TRUE
)

cpmaddpackage("gl:freetype/freetype#VER-2-14-2")

set(IMGUI_INCLUDE_DIR ${imgui_SOURCE_DIR}/ ${imgui_SOURCE_DIR}/backends/)
file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
file(GLOB IMGUI_HEADERS ${imgui_SOURCE_DIR}/*.h)

add_library(imgui STATIC)
target_sources(
    imgui
    PRIVATE ${IMGUI_SOURCES}
            ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
            ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.h
            ${implot_SOURCE_DIR}/implot.cpp
            ${implot_SOURCE_DIR}/implot_items.cpp
            ${implot3d_SOURCE_DIR}/implot3d.cpp
            ${implot3d_SOURCE_DIR}/implot3d_items.cpp
            ${implot3d_SOURCE_DIR}/implot3d_meshes.cpp
)
target_include_directories(
    imgui
    PUBLIC ${IMGUI_INCLUDE_DIR}
           ${implot3d_SOURCE_DIR}
           ${implot_SOURCE_DIR}
           ${imgui-filebrowser_SOURCE_DIR}
)
target_compile_definitions(imgui PUBLIC -DIMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/app_imconfig.h")

target_link_libraries(imgui PRIVATE freetype)

if(APPLE)
    target_sources(
        imgui PRIVATE ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
                      ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    )
    target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLAD_INCLUDE_DIR})
    target_link_libraries(imgui PUBLIC glfw)

elseif(WIN32)
    target_sources(
        imgui PRIVATE ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
                      ${imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp
    )
    target_link_libraries(imgui PUBLIC dxgi d3d12)
else()
    target_sources(
        imgui PRIVATE ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
                      ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )

    target_include_directories(
        imgui
        PUBLIC ${IMGUI_INCLUDE_DIR}
               ${OPENGL_INCLUDE_DIR}
               ${GLFW_INCLUDE_DIR}
               ${GLAD_INCLUDE_DIR}
    )
    target_link_libraries(imgui ${OPENGL_LIBRARIES} glfw)
endif()

cpmaddpackage(
    URI
    "gh:Segfault1602/audio_utils#main"
    OPTIONS
    "AUDIO_UTILS_USE_RTAUDIO ON"
    "AUDIO_UTILS_USE_SNDFILE ON"
    "AUDIO_UTILS_ENABLE_HARDENING ON"
    "AUDIO_UTILS_USE_SANITIZER OFF"
)

cpmaddpackage(
    NAME
    sfFDN
    GIT_REPOSITORY
    https://github.com/Segfault1602/sfFDN.git
    GIT_TAG
    main
)

cpmaddpackage(
    NAME
    FdnOpt
    URI
    "gh:Segfault1602/fdn_opt#main"
)
