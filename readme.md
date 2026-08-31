# Feedback Delay Network Sandbox

This project is a sandbox for experimenting with Feedback Delay Networks (FDNs) for audio processing. It provides a framework for building and testing various FDN configurations and algorithms.

## Build

The library is built using CMake and uses [cpm](https://github.com/cpm-cmake/CPM.cmake) to manage dependencies. CMake presets are provided for building with Ninja+LLVM and MSVC+Visual Studio.

```bash
# configure and build with Ninja and LLVM
 cmake --preset llvm-ninja
 cmake --build --preset llvm --config Debug

# Or, configure with MSVC and Visual Studio
cmake --preset windows

# build
cmake --build --preset windows --config Debug

```

## Dependencies
The following dependencies are required. CPM should take care of all of that for you.
- boost-dll
- boost-math
- Eigen3
- KissFFT
- libSampleRate
- libSndfile
- nanobench
- nlohmann-json
- PFFFT
- quill
- RTAudio
- glfw3
- ImGui
- ImPlot
- ImPlot3D
- imgui-filebrowser
- ensmallen
- Armadillo

These 3 dependencies are also required:
- https://github.com/Segfault1602/fdn_opt
- https://github.com/Segfault1602/sfFDN
- https://github.com/Segfault1602/audio_utils
