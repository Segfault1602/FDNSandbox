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

set(IMGUI_INCLUDE_DIR ${imgui_SOURCE_DIR}/ ${imgui_SOURCE_DIR}/backends/)
file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
file(GLOB IMGUI_HEADERS ${imgui_SOURCE_DIR}/*.h)

add_library(imgui STATIC)
target_sources(
    imgui
    PRIVATE ${IMGUI_SOURCES}
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

if(APPLE)
    add_library(
        imgui STATIC ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
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
    add_library(
        imgui STATIC ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
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
    NAME
    armadillo
    GIT_TAG
    15.2.2
    GIT_REPOSITORY
    https://gitlab.com/conradsnicta/armadillo-code.git
    DOWNLOAD_ONLY
    TRUE
)

if(armadillo_ADDED)
    set(ARMADILLO_INCLUDE_DIR ${armadillo_SOURCE_DIR}/include CACHE PATH "Armadillo include directory")
    add_library(armadillo INTERFACE)
    target_include_directories(armadillo INTERFACE ${armadillo_SOURCE_DIR}/include)
    target_link_libraries(armadillo INTERFACE BLAS::BLAS LAPACK::LAPACK)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(armadillo INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(armadillo INTERFACE ARMA_USE_OPENMP)
    endif()
else()
    message(FATAL_ERROR "Armadillo package not added correctly")
endif()
add_library(Armadillo::Armadillo ALIAS armadillo)

# if(armadillo_ADDED) set(HEADER_ONLY ON CACHE BOOL "Use Armadillo as header-only library" FORCE) add_library(armadillo
# INTERFACE IMPORTED) add_library(Armadillo::Armadillo ALIAS armadillo) target_include_directories(armadillo INTERFACE
# ${armadillo_SOURCE_DIR}/include) set(ARMADILLO_INCLUDE_DIR ${armadillo_SOURCE_DIR}/include CACHE PATH "Armadillo
# include directory") endif()

# if(armadillo_ADDED) add_library(armadillo INTERFACE IMPORTED) target_include_directories(armadillo INTERFACE
# ${armadillo_SOURCE_DIR}/include) if(MKL_FOUND) target_compile_definitions(armadillo INTERFACE ARMA_USE_MKL)
# target_link_libraries(armadillo INTERFACE MKL::MKL) endif() endif() add_library(Armadillo::Armadillo alias armadillo)

cpmaddpackage(
    NAME
    ensmallen
    GIT_REPOSITORY
    https://github.com/mlpack/ensmallen.git
    GIT_TAG
    master
    DOWNLOAD_ONLY
    TRUE
)

if(ensmallen_ADDED)
    add_library(ensmallen INTERFACE IMPORTED)
    target_include_directories(ensmallen INTERFACE ${ensmallen_SOURCE_DIR}/include)
    set(ENSMALLEN_INCLUDE_DIR ${ensmallen_SOURCE_DIR}/include CACHE PATH "Ensmallen include directory")
else()
    message(FATAL_ERROR "Ensmallen package not added correctly")
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
