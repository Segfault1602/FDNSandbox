#pragma once

#include "audio/audio_clip.h"
#include "audio/audio_events.h"
#include "audio/audio_runtime.h"
#include "audio/audio_types.h"

#include <audio_utils/audio_manager.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace fdn_sandbox::audio
{

struct AudioEngineConfig
{
    std::uint32_t sample_rate;
};

class AudioEngine final
{
  public:
    explicit AudioEngine(AudioEngineConfig config);
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) = delete;
    AudioEngine& operator=(AudioEngine&&) = delete;

    [[nodiscard]] bool Start();
    void Stop() noexcept;
    void Shutdown() noexcept;
    [[nodiscard]] bool IsRunning() const noexcept;

    [[nodiscard]] bool TryInstallFdn(std::uint64_t generation, std::unique_ptr<sfFDN::FDN> value);
    [[nodiscard]] bool TryInstallConvolver(std::unique_ptr<sfFDN::PartitionedConvolver> value);
    [[nodiscard]] bool TryInstallClip(std::unique_ptr<AudioClip> value);
    [[nodiscard]] bool PlayClip(bool loop);
    [[nodiscard]] bool StopClip();
    [[nodiscard]] bool TriggerImpulse();

    void FlushPendingCommands();
    void CollectRetiredObjects();
    std::vector<AudioEvent> PollEvents();

    void SetOutputGain(float value) noexcept;
    void SetFdnMix(float wet, float dry) noexcept;
    void SetConvolutionWet(float value) noexcept;
    void SetDirectDelayMs(std::uint32_t value) noexcept;
    void SetReverbEngine(ReverbEngine value) noexcept;
    ReverbEngine GetReverbEngine() const noexcept;
    PlaybackState GetPlaybackState() const noexcept;

    AudioTelemetry ReadTelemetry() const noexcept;
    float ConsumePeak() noexcept;
    bool ConsumeClipped() noexcept;

    std::vector<std::string> GetSupportedAudioDrivers() const;
    std::string GetCurrentAudioDriver() const;
    void SetAudioDriver(std::string_view driver);
    std::vector<std::string> GetOutputDevices() const;
    void SetOutputDevice(std::string_view device);
    audio_stream_info GetStreamInfo() const;
    void PlayTestTone(bool play);

  private:
    void AudioCallback(std::span<float> output_buffer, std::size_t frame_size, std::size_t num_channels) noexcept;

    AudioRuntime runtime_;
    std::unique_ptr<audio_manager> manager_;
    bool shutdown_ = false;
};

} // namespace fdn_sandbox::audio
