#pragma once

#include <cstddef>
#include <cstdint>

namespace fdn_sandbox::audio
{

inline constexpr std::size_t kSystemBlockSize = 1024;
inline constexpr float kMeterIntegrationSeconds = 0.3f;

enum class ReverbEngine : std::uint8_t
{
    Fdn,
    Convolution,
};

enum class PlaybackState : std::uint8_t
{
    Stopped,
    Playing,
};

struct AudioTelemetry
{
    float rms = 0.0f;
    float cpu_usage = 0.0f;
    PlaybackState playback_state = PlaybackState::Stopped;
    std::uint64_t active_fdn_generation = 0;
    std::uint64_t command_queue_full_count = 0;
    std::uint64_t coalesced_command_count = 0;
    std::uint64_t event_overflow_count = 0;
};

} // namespace fdn_sandbox::audio
