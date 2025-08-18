set(APP_SOURCES main.cpp utils.cpp widget.cpp app.cpp fdn_info.cpp)

add_subdirectory(analysis)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE imgui
          implot
          ${OPENGL_LIBRARIES}
          audio_utils
          fdn_analysis
          sfFDN
          SndFile::sndfile
          Boost::math
          Eigen3::Eigen)

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${IMGUI_FILEBROWSER_INCLUDE_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(
  ${PROJECT_NAME} PUBLIC GL_SILENCE_DEPRECATION
                         -DIMGUI_USER_CONFIG="app_imconfig.h")

target_compile_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_CXX_COMPILE_OPTIONS})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_link_options(${PROJECT_NAME} PUBLIC ${FDNSANDBOX_LINK_OPTIONS})
