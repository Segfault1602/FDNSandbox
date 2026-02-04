add_subdirectory(optimization)

# ====== FDN Sandbox App logic =======
add_library(fdn_sandbox_app STATIC)
target_sources(
    fdn_sandbox_app
    PRIVATE utils.cpp
            widget.cpp
            app.cpp
            fdn_analyzer.cpp
            ir_analyzer.cpp
            fdn_info.cpp
            presets.cpp
            app_imconfig.h
            optimization_gui.cpp
)
target_link_libraries(
    fdn_sandbox_app
    PUBLIC audio_utils
           quill::quill
           sfFDN.sfFDN
           imgui
           fdn_optimization
           SndFile::sndfile
           Boost::math
           Boost::filesystem
           Boost::dll
           Eigen3::Eigen
           nlohmann_json::nlohmann_json
           fdn_sandbox::fdn_sandbox_options
           fdn_sandbox::fdn_sandbox_warnings
)

target_compile_features(fdn_sandbox_app PRIVATE cxx_std_23)
target_compile_definitions(fdn_sandbox_app PUBLIC -DIMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/app_imconfig.h")

# ====== Metal implementation ======
if(APPLE)
    add_executable(${PROJECT_NAME})
    target_sources(${PROJECT_NAME} PRIVATE main.mm)
    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE imgui
                fdn_sandbox_app
                "-framework Metal"
                "-framework MetalKit"
                "-framework Cocoa"
                "-framework IOKit"
                "-framework CoreVideo"
                "-framework QuartzCore"
    )
elseif(WIN32)
    add_executable(${PROJECT_NAME} main_win32_dx12.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE fdn_sandbox_app imgui)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DIMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/src/app_imconfig.h")

else()
    # ========= OpenGL implementation =========
    add_executable(${PROJECT_NAME} main.cpp)

    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE fdn_sandbox_app
                imgui
                implot
                ${OPENGL_LIBRARIES}
    )

    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_definitions(${PROJECT_NAME} PUBLIC GL_SILENCE_DEPRECATION)
endif()
target_compile_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_CXX_COMPILE_OPTIONS})
target_link_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_LINK_OPTIONS})

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../audio
                                              $<TARGET_FILE_DIR:${PROJECT_NAME}>/audio
)
