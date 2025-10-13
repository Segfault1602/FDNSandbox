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

FetchContent_MakeAvailable(imgui)

set(IMGUI_INCLUDE_DIR ${imgui_SOURCE_DIR}/ ${imgui_SOURCE_DIR}/backends/)
file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
file(GLOB IMGUI_HEADERS ${imgui_SOURCE_DIR}/*.h)
add_library(
    imgui STATIC
    ${IMGUI_SOURCES}
    ${IMGUI_SOURCES}
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
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

if (APPLE)
add_library(
    imgui_metal STATIC
    ${IMGUI_SOURCES}
    ${IMGUI_SOURCES}
    ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)

target_include_directories(imgui_metal PUBLIC ${IMGUI_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLAD_INCLUDE_DIR})
target_link_libraries(imgui_metal glfw)
endif()


FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master)
FetchContent_MakeAvailable(implot)

set(IMPLOT_INCLUDE_DIR ${implot_SOURCE_DIR})
file(GLOB IMPLOT_SOURCES ${implot_SOURCE_DIR}/*.cpp)
add_library(implot STATIC ${IMPLOT_SOURCES})
target_include_directories(implot PUBLIC ${IMPLOT_INCLUDE_DIR} ${IMGUI_INCLUDE_DIR})

FetchContent_Declare(imgui-filebrowser GIT_REPOSITORY https://github.com/AirGuanZ/imgui-filebrowser.git GIT_TAG master)

FetchContent_MakeAvailable(imgui-filebrowser)
set(IMGUI_FILEBROWSER_INCLUDE_DIR ${imgui-filebrowser_SOURCE_DIR})
