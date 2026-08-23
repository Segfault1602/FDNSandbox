#pragma once

#include "audio/audio_block_ops.h"
#include "audio/audio_events.h"
#include "audio/audio_handoff.h"
#include "audio/audio_types.h"

#include <sffdn/sffdn.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace fdn_sandbox::audio
{

class AudioRuntime final
{
  public:
    explicit AudioRuntime(std::uint32_t sample_rate, std::size_t queue_capacity = 16,
                          std::size_t max_commands_per_block = 4);
    ~AudioRuntime();

    AudioRuntime(const AudioRuntime&) = delete;
    AudioRuntime& operator=(const AudioRuntime&) = delete;
    AudioRuntime(AudioRuntime&&) = delete;
    AudioRuntime& operator=(AudioRuntime&&) = delete;

    std::span<float> PrepareInput(std::size_t frame_size) noexcept;
    void Process(std::span<float> output_buffer, std::size_t frame_size, std::size_t num_channels) noexcept;

    bool SubmitFdn(std::uint64_t generation, std::unique_ptr<sfFDN::FDN> value);
    bool SubmitConvolver(std::unique_ptr<sfFDN::PartitionedConvolver> value);
    bool SubmitClip(std::unique_ptr<AudioClip> value);
    bool PlayClip(bool loop);
    bool StopClip();
    bool TriggerImpulse();
    void FlushPendingCommands();
    std::size_t CollectRetiredObjects();
    bool TryPopEvent(AudioEvent& event) noexcept;
    bool HasActiveFdn() const noexcept;
    PlaybackState GetPlaybackState() const noexcept;
    void StopAcceptingCommands() noexcept;
    // The audio callback must be stopped before this is called.
    void DrainOffRealtime();

    void SetOutputGain(float value) noexcept;
    void SetFdnMix(float wet, float dry) noexcept;
    void SetConvolutionWet(float value) noexcept;
    void SetDirectDelayMs(std::uint32_t value) noexcept;
    void SetReverbEngine(ReverbEngine value) noexcept;
    ReverbEngine GetReverbEngine() const noexcept;

    AudioTelemetry ReadTelemetry() const noexcept;
    float ConsumePeak() noexcept;
    bool ConsumeClipped() noexcept;

  private:
    struct ProcessingTimes
    {
        std::int64_t convolution_ns = 0;
        std::int64_t fdn_ns = 0;
    };

    bool ApplyCommand(AudioCommand& command) noexcept;
    void ProcessCommands() noexcept;
    void RenderClip() noexcept;
    ProcessingTimes ProcessEngines() noexcept;
    void DetectInstability() noexcept;
    void PrepareMix(ReverbEngine engine) noexcept;
    void MixMeterAndWrite(std::span<float> output_buffer, std::span<float> output, std::size_t num_channels,
                          float wet, float dry) noexcept;
    void UpdateCpuUsage(std::int64_t duration_ns) noexcept;
    void PublishEvent(AudioEvent event) noexcept;

    std::uint32_t sample_rate_;
    std::size_t max_commands_per_block_;
    AudioHandoff handoff_;
    std::optional<AudioCommand> deferred_command_;

    std::unique_ptr<sfFDN::FDN> active_fdn_;
    std::unique_ptr<sfFDN::PartitionedConvolver> active_convolver_;
    std::unique_ptr<AudioClip> active_clip_;
    std::size_t clip_cursor_ = 0;
    bool clip_loop_ = false;
    std::uint64_t active_fdn_generation_ = 0;
    bool impulse_requested_ = false;
    bool impulse_active_ = false;
    std::size_t last_invalid_frame_size_ = 0;
    ReverbEngine last_reverb_engine_ = ReverbEngine::Fdn;
    std::size_t cpu_highwater_mark_ns_ = 0;

    std::array<float, kSystemBlockSize> input_{};
    std::array<float, kSystemBlockSize> fdn_output_{};
    std::array<float, kSystemBlockSize> convolution_output_{};
    sfFDN::Delay direct_delay_;
    OutputLevelMeter output_meter_;
    CpuAverage cpu_average_;

    std::atomic<float> output_gain_ = 1.0f;
    std::atomic<float> fdn_dry_ = 0.5f;
    std::atomic<float> fdn_wet_ = 0.5f;
    std::atomic<float> convolution_wet_ = 1.0f;
    std::atomic<std::uint32_t> direct_delay_ms_ = 0;
    std::atomic<ReverbEngine> reverb_engine_ = ReverbEngine::Fdn;

    std::atomic<float> meter_rms_ = 0.0f;
    std::atomic<float> meter_peak_ = 0.0f;
    std::atomic<bool> meter_clipped_ = false;
    std::atomic<float> cpu_usage_ = 0.0f;
    std::atomic<PlaybackState> playback_state_ = PlaybackState::Stopped;
    std::atomic<std::uint64_t> telemetry_generation_ = 0;
    std::atomic<std::uint64_t> event_overflow_count_ = 0;
};

} // namespace fdn_sandbox::audio
