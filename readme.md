# Feedback Delay Network Sandbox

This project is a sandbox for experimenting with Feedback Delay Networks (FDNs) for audio processing. It provides a framework for building and testing various FDN configurations and algorithms.

## Build

The library is built using CMake and uses [cpm](https://github.com/cpm-cmake/CPM.cmake) to manage dependencies. CMake presets are provided for building with Ninja+LLVM and MSVC+Visual Studio.

```bash
# configure with Ninja and LLVM
cmake --preset llvm-ninja

# build
cmake --build --preset llvm-debug

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

If you use CMakePreset.json, CMake will expect these repository to be already checked in on your machine at the path defined by the `CPM_##_SOURCE` variable. You can simply delete those variables if you want CPM to download those for you instead.
