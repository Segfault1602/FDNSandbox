# ====== Headless core logic ======
add_library(fdn_sandbox_core STATIC presets.cpp session/fdn_session.cpp)
add_library(fdn_sandbox::core ALIAS fdn_sandbox_core)

target_include_directories(fdn_sandbox_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    fdn_sandbox_core
    PUBLIC sfFDN::sfFDN
           fdn_sandbox::fdn_sandbox_options
    PRIVATE quill::quill
            fdn_sandbox::fdn_sandbox_warnings
)

# ====== Headless analysis logic ======
add_library(fdn_sandbox_analysis STATIC analysis/analysis_workspace.cpp fdn_analyzer.cpp ir_analyzer.cpp)
add_library(fdn_sandbox::analysis ALIAS fdn_sandbox_analysis)

target_include_directories(fdn_sandbox_analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    fdn_sandbox_analysis
    PUBLIC audio_utils::audio_utils
           quill::quill
           sfFDN::sfFDN
           fdn_sandbox::fdn_sandbox_options
    PRIVATE fdn_sandbox::fdn_sandbox_warnings
)

# ====== Headless audio logic ======
add_library(
    fdn_sandbox_audio
    STATIC audio/audio_block_ops.cpp
           audio/audio_runtime.cpp
)
add_library(fdn_sandbox::audio ALIAS fdn_sandbox_audio)

target_include_directories(fdn_sandbox_audio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    fdn_sandbox_audio
    PUBLIC fdn_sandbox::fdn_sandbox_options
           readerwriterqueue
           sfFDN::sfFDN
    PRIVATE Eigen3::Eigen
            samplerate
            SndFile::sndfile
            fdn_sandbox::fdn_sandbox_warnings
)

# ====== Concrete audio device engine ======
add_library(fdn_sandbox_audio_engine STATIC audio/audio_engine.cpp)
add_library(fdn_sandbox::audio_engine ALIAS fdn_sandbox_audio_engine)

target_include_directories(fdn_sandbox_audio_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    fdn_sandbox_audio_engine
    PUBLIC audio_utils::audio_utils
           fdn_sandbox::audio
           fdn_sandbox::fdn_sandbox_options
    PRIVATE fdn_sandbox::fdn_sandbox_warnings
)

# ====== Headless file workflows ======
add_library(fdn_sandbox_files STATIC files/file_workflows.cpp)
add_library(fdn_sandbox::files ALIAS fdn_sandbox_files)

target_include_directories(fdn_sandbox_files PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    fdn_sandbox_files
    PUBLIC sfFDN::sfFDN
           fdn_sandbox::fdn_sandbox_options
    PRIVATE samplerate
            SndFile::sndfile
            fdn_sandbox::fdn_sandbox_warnings
)

# ====== ImGui views ======
add_library(fdn_sandbox_ui STATIC)
target_sources(
    fdn_sandbox_ui
    PRIVATE utils.cpp
            widget.cpp
            files/file_dialogs.cpp
            fdn_info.cpp
            fdn_widget.cpp
            app_imconfig.h
            icons.h
            plot_ui.cpp
            plot_ui.h
            theme.cpp
            theme.h
            views/analysis_views.cpp
            views/audio_view.cpp
            views/fdn_configurator_view.cpp
            views/fdn_info_view.cpp
            views/optimization_view.cpp
            views/settings_view.cpp
            views/shell_view.cpp
            views/window_ids.h
)
add_library(fdn_sandbox::ui ALIAS fdn_sandbox_ui)

target_link_libraries(
    fdn_sandbox_ui
    PUBLIC audio_utils::audio_utils
           quill::quill
           sfFDN::sfFDN
           imgui
           fdn_optimization
           SndFile::sndfile
           Boost::math
           Boost::filesystem
           Boost::dll
           Eigen3::Eigen
           nlohmann_json::nlohmann_json
           fdn_sandbox::core
           fdn_sandbox::analysis
           fdn_sandbox::audio
           fdn_sandbox::audio_engine
           fdn_sandbox::fdn_sandbox_options
    PRIVATE fdn_sandbox::fdn_sandbox_warnings
)
target_include_directories(fdn_sandbox_ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${IconFontCppHeaders_SOURCE_DIR})

# ====== FDN Sandbox App orchestration =======
add_library(
    fdn_sandbox_app
    STATIC app.cpp
           app_notifications.cpp
           app_menu.cpp
           app_audio.cpp
           notifications.cpp
           notifications.h
)
target_link_libraries(
    fdn_sandbox_app
    PUBLIC fdn_sandbox::ui
           fdn_sandbox::files
           fdn_sandbox::fdn_sandbox_options
    PRIVATE Boost::dll
            fdn_sandbox::fdn_sandbox_warnings
)
target_include_directories(fdn_sandbox_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ====== Metal implementation ======
if(APPLE)
    add_executable(${PROJECT_NAME})
    target_sources(${PROJECT_NAME} PRIVATE main.mm)
    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE imgui
                OpenMP::OpenMP_CXX
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
    target_link_libraries(${PROJECT_NAME} PRIVATE fdn_sandbox_app imgui freetype)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

else()
    # ========= OpenGL implementation =========
    find_package(TBB REQUIRED)
    add_executable(${PROJECT_NAME} main.cpp)

    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE fdn_sandbox_app
                imgui
                tbb
                ${OPENGL_LIBRARIES}
    )

    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_definitions(${PROJECT_NAME} PUBLIC GL_SILENCE_DEPRECATION)
endif()
target_compile_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_CXX_COMPILE_OPTIONS})
target_link_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_LINK_OPTIONS})

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../audio
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/audio
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../fonts
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${fontaudio_SOURCE_DIR}/font/fontaudio.ttf
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/fonts/fontaudio.ttf
)
