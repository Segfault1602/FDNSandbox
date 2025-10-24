find_package(OpenMP)

add_library(fdn_analysis STATIC)
target_sources(
    fdn_analysis
    PRIVATE analysis.cpp fdn_analyzer.cpp ir_analyzer.cpp
    PUBLIC FILE_SET
           "fdn_analysis_headers"
           TYPE
           HEADERS
           FILES
           analysis.h
           fdn_analyzer.h
)

target_link_libraries(
    fdn_analysis
    PRIVATE Eigen3::Eigen
            Boost::math
            audio_utils::audio_utils
            SndFile::sndfile
            sfFDN.sfFDN
            quill::quill
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fdn_analysis PRIVATE OpenMP::OpenMP_CXX)
endif()
