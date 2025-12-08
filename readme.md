# Feedback Delay Network Sandbox

This project is a sandbox for experimenting with Feedback Delay Networks (FDNs) for audio processing. It provides a framework for building and testing various FDN configurations and algorithms.

## Build

The library is built using CMake and uses [vcpkg](https://vcpkg.io/en/) to manage dependencies. CMake presets are provided for building with Ninja+LLVM and MSVC+Visual Studio.

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

### Submodules

Make sure to initialize and update the git submodules:

```bash
git submodule update --init --recursive
```

- [audio_utils](https://github.com/Segfault1602/audio_utils) - Audio I/O and DSP utilities
- [sfFDN](https://github.com/Segfault1602/sfFDN) - Feedback Delay Network library

### vcpkg Packages
The following vcpkg packages are required:
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

### CMake FetchContent
The following dependencies are managed using CMake's FetchContent:
- ImGui
- ImPlot
- imgui-filebrowser