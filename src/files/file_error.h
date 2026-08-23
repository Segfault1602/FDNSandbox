#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace fdn_sandbox::files
{

enum class FileErrorKind : std::uint8_t
{
    InvalidArgument,
    OpenFailed,
    InvalidMetadata,
    DecodeFailed,
    UnsupportedChannelCount,
    EmptyAudio,
    ResampleFailed,
    WriteFailed,
    FinalizeFailed,
    ConvolverBuildFailed
};

/// Stable category, affected path, and diagnostic detail for a failed file operation.
struct FileError
{
    FileErrorKind kind;
    std::filesystem::path path;
    std::string message;
};

} // namespace fdn_sandbox::files
