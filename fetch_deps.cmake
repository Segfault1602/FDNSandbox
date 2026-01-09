find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS math dll)
find_package(
    Eigen3
    3.4
    REQUIRED
    NO_MODULE
)
find_package(SndFile REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(quill CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG docking)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master)
FetchContent_Declare(implot3d GIT_REPOSITORY https://github.com/brenocq/implot3d.git GIT_TAG main)
FetchContent_MakeAvailable(imgui implot implot3d)

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
target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${implot3d_SOURCE_DIR} ${implot_SOURCE_DIR})
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

# set(IMPLOT_INCLUDE_DIR ${implot_SOURCE_DIR}) set(IMPLOT_SOURCES ${implot_SOURCE_DIR}/implot.cpp
# ${implot_SOURCE_DIR}/implot_items.cpp)

# target_source( imgui PRIVATE ${implot_SOURCE_DIR}/implot.cpp ${implot_SOURCE_DIR}/implot_items.cpp )

# add_library(implot STATIC ${IMPLOT_SOURCES}) target_include_directories(implot PUBLIC ${IMPLOT_INCLUDE_DIR}
# ${IMGUI_INCLUDE_DIR}) target_compile_definitions(implot PUBLIC
# -DIMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/app_imconfig.h")

# set(IMPLOT3D_INCLUDE_DIR ${implot3d_SOURCE_DIR}) file(GLOB IMPLOT3D_SOURCES ${implot3d_SOURCE_DIR}/*.cpp)
# add_library(implot3d STATIC ${IMPLOT3D_SOURCES}) target_include_directories(implot3d PUBLIC ${IMPLOT3D_INCLUDE_DIR}
# ${IMGUI_INCLUDE_DIR}) target_compile_definitions(implot3d PUBLIC
# -DIMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/app_imconfig.h")

FetchContent_Declare(imgui-filebrowser GIT_REPOSITORY https://github.com/AirGuanZ/imgui-filebrowser.git GIT_TAG master)

FetchContent_MakeAvailable(imgui-filebrowser)
set(IMGUI_FILEBROWSER_INCLUDE_DIR ${imgui-filebrowser_SOURCE_DIR})
