#pragma once

#include <cstddef>
#include <cstdint>

namespace fdn_sandbox::audio
{

// Block size requested from the audio device. Devices are free to ignore this hint, so the runtime
// must accept whatever the callback actually delivers.
inline constexpr std::size_t kSystemBlockSize = 1024;
// Largest device block the runtime can process in a single pass. Larger callbacks are split into
// several passes, so this only bounds the internal scratch buffers.
inline constexpr std::size_t kMaxBlockSize = 8192;
// Block size used by the partitioned convolution engine. It must be a power of two, so it cannot
// simply follow the device block size; mismatched device blocks are bridged by a fixed-block adapter.
inline constexpr std::size_t kConvolutionBlockSize = 1024;
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
