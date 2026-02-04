add_library(fdn_optimization STATIC)
target_sources(
    fdn_optimization
    PRIVATE optimizer.cpp
            model.cpp
            loss.cpp
            random_searcher.cpp
    PUBLIC FILE_SET
           "fdn_optimization_headers"
           TYPE
           HEADERS
           FILES
           optimizer.h
           model.h
           loss.h
)

target_link_libraries(
    fdn_optimization
    PRIVATE audio_utils::audio_utils
            sfFDN.sfFDN
            quill::quill
            Armadillo::Armadillo
            ensmallen
            fdn_sandbox::fdn_sandbox_options
            fdn_sandbox::fdn_sandbox_warnings
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fdn_optimization PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(fdn_optimization SYSTEM PRIVATE ${ensmallen_SOURCE_DIR}/include)

add_executable(optim_tool main.cpp)
target_link_libraries(
    optim_tool
    PRIVATE fdn_optimization
            sfFDN.sfFDN
            quill::quill
            audio_utils::audio_utils
            Armadillo::Armadillo
            fdn_sandbox::fdn_sandbox_options
            fdn_sandbox::fdn_sandbox_warnings
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(optim_tool PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(tuning tuning.cpp)
target_link_libraries(
    tuning
    PRIVATE fdn_optimization
            sfFDN.sfFDN
            quill::quill
            audio_utils::audio_utils
            Armadillo::Armadillo
            fdn_sandbox::fdn_sandbox_options
            fdn_sandbox::fdn_sandbox_warnings
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tuning PRIVATE OpenMP::OpenMP_CXX)
endif()
