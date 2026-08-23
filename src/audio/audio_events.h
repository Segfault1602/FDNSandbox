#pragma once

#include <cstdint>

namespace fdn_sandbox::audio
{

enum class AudioEventKind : std::uint8_t
{
    FrameSizeMismatch,
    FdnInstability,
    ConvolverBlockSizeMismatch,
    DirectDelayOutOfRange,
    CpuHighWater,
    ClipFinished,
};

struct AudioEvent
{
    AudioEventKind kind;
    std::uint64_t generation = 0;
    std::uint64_t actual = 0;
    std::uint64_t expected = 0;
    float value = 0.0f;
};

} // namespace fdn_sandbox::audio
