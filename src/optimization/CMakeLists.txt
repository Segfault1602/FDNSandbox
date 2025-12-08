find_package(OpenMP)
find_package(Armadillo CONFIG REQUIRED)
add_library(Armadillo::Armadillo ALIAS armadillo)

include(FetchContent)
FetchContent_Declare(ensmallen GIT_REPOSITORY https://github.com/mlpack/ensmallen.git GIT_TAG master)
set(USE_BANDICOOT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ensmallen)

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
            fdn_analysis
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fdn_optimization PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(fdn_optimization SYSTEM PRIVATE ${ensmallen_SOURCE_DIR}/include)

target_compile_options(fdn_optimization PRIVATE ${FDNSANDBOX_CXX_COMPILE_OPTIONS})
target_link_options(fdn_optimization PUBLIC ${FDNSANDBOX_LINK_OPTIONS})
target_compile_definitions(fdn_optimization PUBLIC ${FDNSANDBOX_COMPILE_DEFINITION})

add_executable(optim_tool main.cpp)
target_link_libraries(
    optim_tool
    PRIVATE fdn_optimization
            sfFDN.sfFDN
            quill::quill
            audio_utils::audio_utils
)
target_compile_options(optim_tool PRIVATE ${FDNSANDBOX_CXX_COMPILE_OPTIONS})
target_link_options(optim_tool PUBLIC ${FDNSANDBOX_LINK_OPTIONS})
target_compile_definitions(optim_tool PUBLIC ${FDNSANDBOX_COMPILE_DEFINITION})
