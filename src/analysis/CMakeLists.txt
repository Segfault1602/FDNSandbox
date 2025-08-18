add_library(fdn_analysis STATIC)
target_sources(
  fdn_analysis
  PRIVATE analysis.cpp fdn_analyzer.cpp
  PUBLIC FILE_SET
         "fdn_analysis_headers"
         TYPE
         HEADERS
         FILES
         analysis.h
         fdn_analyzer.h)

target_link_libraries(fdn_analysis PRIVATE Eigen3::Eigen Boost::math
                                           audio_utils::audio_utils sfFDN)
