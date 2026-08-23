#include "audio/audio_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <type_traits>
#include <utility>
#include <variant>

namespace fdn_sandbox::audio
{

AudioRuntime::AudioRuntime(std::uint32_t sample_rate, std::size_t queue_capacity, std::size_t max_commands_per_block)
    : sample_rate_(sample_rate)
    , max_commands_per_block_(max_commands_per_block)
    , handoff_(queue_capacity)
    , direct_delay_(0, sample_rate)
    , output_meter_(static_cast<std::size_t>(static_cast<float>(sample_rate) * kMeterIntegrationSeconds))
{
}

AudioRuntime::~AudioRuntime()
{
    DrainOffRealtime();
}

std::span<float> AudioRuntime::PrepareInput(std::size_t frame_size) noexcept
{
    ProcessCommands();
    if (frame_size != kSystemBlockSize)
    {
        if (frame_size != last_invalid_frame_size_)
        {
            PublishEvent({
                .kind = AudioEventKind::FrameSizeMismatch,
                .actual = frame_size,
                .expected = kSystemBlockSize,
            });
            last_invalid_frame_size_ = frame_size;
        }
        return {};
    }
    last_invalid_frame_size_ = 0;

    input_.fill(0.0f);
    fdn_output_.fill(0.0f);
    convolution_output_.fill(0.0f);

    impulse_active_ = active_fdn_ != nullptr && impulse_requested_;
    if (impulse_active_)
    {
        input_.front() = 1.0f;
    }
    if (active_fdn_ != nullptr)
    {
        RenderClip();
    }
    return input_;
}

void AudioRuntime::Process(std::span<float> output_buffer, std::size_t frame_size, std::size_t num_channels) noexcept
{
    if (frame_size != kSystemBlockSize || output_buffer.size() < frame_size * num_channels)
    {
        return;
    }
    if (active_fdn_ == nullptr)
    {
        std::ranges::fill(output_buffer, 0.0f);
        return;
    }

    active_fdn_->SetDirectGain(0.0f);
    const ProcessingTimes processing_times = ProcessEngines();
    DetectInstability();

    const ReverbEngine reverb_engine = reverb_engine_.load(std::memory_order_relaxed);
    if (last_reverb_engine_ == ReverbEngine::Fdn && reverb_engine == ReverbEngine::Convolution)
    {
        Crossfade(convolution_output_, fdn_output_, convolution_output_);
    }
    else if (last_reverb_engine_ == ReverbEngine::Convolution && reverb_engine == ReverbEngine::Fdn)
    {
        Crossfade(fdn_output_, convolution_output_, fdn_output_);
    }
    last_reverb_engine_ = reverb_engine;

    if (impulse_active_)
    {
        input_.front() -= 1.0f;
        impulse_requested_ = false;
        impulse_active_ = false;
    }

    PrepareMix(reverb_engine);
    std::span<float> output =
        reverb_engine == ReverbEngine::Fdn ? std::span<float>(fdn_output_) : std::span<float>(convolution_output_);
    const float wet = reverb_engine == ReverbEngine::Fdn ? fdn_wet_.load(std::memory_order_relaxed)
                                                         : convolution_wet_.load(std::memory_order_relaxed);
    const float dry = reverb_engine == ReverbEngine::Fdn ? fdn_dry_.load(std::memory_order_relaxed) : 0.0f;
    MixMeterAndWrite(output_buffer, output, num_channels, wet, dry);

    const std::int64_t duration =
        reverb_engine == ReverbEngine::Fdn ? processing_times.fdn_ns : processing_times.convolution_ns;
    UpdateCpuUsage(duration);
}

bool AudioRuntime::SubmitFdn(std::uint64_t generation, std::unique_ptr<sfFDN::FDN> value)
{
    return handoff_.SubmitFdn(generation, std::move(value));
}

bool AudioRuntime::SubmitConvolver(std::unique_ptr<sfFDN::PartitionedConvolver> value)
{
    return handoff_.SubmitConvolver(std::move(value));
}

bool AudioRuntime::SubmitClip(std::unique_ptr<AudioClip> value)
{
    return handoff_.SubmitClip(std::move(value));
}

bool AudioRuntime::PlayClip(bool loop)
{
    return handoff_.SubmitPlay(loop);
}

bool AudioRuntime::StopClip()
{
    return handoff_.SubmitStop();
}

bool AudioRuntime::TriggerImpulse()
{
    return handoff_.SubmitImpulse();
}

void AudioRuntime::FlushPendingCommands()
{
    handoff_.FlushPendingCommands();
}

std::size_t AudioRuntime::CollectRetiredObjects()
{
    return handoff_.CollectRetiredObjects();
}

bool AudioRuntime::TryPopEvent(AudioEvent& event) noexcept
{
    return handoff_.TryPopEvent(event);
}

PlaybackState AudioRuntime::GetPlaybackState() const noexcept
{
    return playback_state_.load(std::memory_order_relaxed);
}

void AudioRuntime::StopAcceptingCommands() noexcept
{
    handoff_.StopAcceptingCommands();
}

void AudioRuntime::DrainOffRealtime()
{
    deferred_command_.reset();
    active_fdn_.reset();
    active_convolver_.reset();
    active_clip_.reset();
    handoff_.DrainOffRealtime();
}

void AudioRuntime::SetOutputGain(float value) noexcept
{
    output_gain_.store(value, std::memory_order_relaxed);
}

void AudioRuntime::SetFdnMix(float wet, float dry) noexcept
{
    fdn_wet_.store(wet, std::memory_order_relaxed);
    fdn_dry_.store(dry, std::memory_order_relaxed);
}

void AudioRuntime::SetConvolutionWet(float value) noexcept
{
    convolution_wet_.store(value, std::memory_order_relaxed);
}

void AudioRuntime::SetDirectDelayMs(std::uint32_t value) noexcept
{
    direct_delay_ms_.store(value, std::memory_order_relaxed);
}

void AudioRuntime::SetReverbEngine(ReverbEngine value) noexcept
{
    reverb_engine_.store(value, std::memory_order_relaxed);
}

ReverbEngine AudioRuntime::GetReverbEngine() const noexcept
{
    return reverb_engine_.load(std::memory_order_relaxed);
}

AudioTelemetry AudioRuntime::ReadTelemetry() const noexcept
{
    return {
        .rms = meter_rms_.load(std::memory_order_relaxed),
        .cpu_usage = cpu_usage_.load(std::memory_order_relaxed),
        .playback_state = playback_state_.load(std::memory_order_relaxed),
        .active_fdn_generation = telemetry_generation_.load(std::memory_order_relaxed),
        .command_queue_full_count = handoff_.CommandQueueFullCount(),
        .coalesced_command_count = handoff_.CoalescedCommandCount(),
        .event_overflow_count = event_overflow_count_.load(std::memory_order_relaxed),
    };
}

float AudioRuntime::ConsumePeak() noexcept
{
    return meter_peak_.exchange(0.0f, std::memory_order_relaxed);
}

bool AudioRuntime::ConsumeClipped() noexcept
{
    return meter_clipped_.exchange(false, std::memory_order_relaxed);
}

bool AudioRuntime::ApplyCommand(AudioCommand& command) noexcept
{
    return std::visit(
        [this](auto& value) -> bool {
            using Command = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Command, InstallFdn>)
            {
                if (value.value == nullptr)
                {
                    return true;
                }
                if (!handoff_.TryRetire(active_fdn_))
                {
                    return false;
                }
                active_fdn_ = std::move(value.value);
                active_fdn_generation_ = value.generation;
                telemetry_generation_.store(value.generation, std::memory_order_relaxed);
                return true;
            }
            else if constexpr (std::is_same_v<Command, InstallConvolver>)
            {
                if (value.value == nullptr)
                {
                    return true;
                }
                if (value.value->GetBlockSize() != kSystemBlockSize)
                {
                    if (!handoff_.TryRetire(active_convolver_))
                    {
                        return false;
                    }
                    const std::size_t actual_block_size = value.value->GetBlockSize();
                    if (!handoff_.TryRetire(value.value))
                    {
                        return false;
                    }
                    PublishEvent({
                        .kind = AudioEventKind::ConvolverBlockSizeMismatch,
                        .actual = actual_block_size,
                        .expected = kSystemBlockSize,
                    });
                    return true;
                }
                if (!handoff_.TryRetire(active_convolver_))
                {
                    return false;
                }
                active_convolver_ = std::move(value.value);
                return true;
            }
            else if constexpr (std::is_same_v<Command, InstallClip>)
            {
                if (value.value == nullptr)
                {
                    return true;
                }
                if (!handoff_.TryRetire(active_clip_))
                {
                    return false;
                }
                active_clip_ = std::move(value.value);
                clip_cursor_ = 0;
                playback_state_.store(PlaybackState::Stopped, std::memory_order_relaxed);
                return true;
            }
            else if constexpr (std::is_same_v<Command, fdn_sandbox::audio::PlayClip>)
            {
                if (active_clip_ != nullptr)
                {
                    clip_loop_ = value.loop;
                    playback_state_.store(PlaybackState::Playing, std::memory_order_relaxed);
                }
                return true;
            }
            else if constexpr (std::is_same_v<Command, fdn_sandbox::audio::StopClip>)
            {
                clip_cursor_ = 0;
                playback_state_.store(PlaybackState::Stopped, std::memory_order_relaxed);
                return true;
            }
            else
            {
                impulse_requested_ = true;
                return true;
            }
        },
        command);
}

void AudioRuntime::ProcessCommands() noexcept
{
    if (deferred_command_)
    {
        if (!ApplyCommand(*deferred_command_))
        {
            return;
        }
        deferred_command_.reset();
    }

    for (std::size_t processed = 0; processed < max_commands_per_block_; ++processed)
    {
        AudioCommand command{fdn_sandbox::audio::TriggerImpulse{}};
        if (!handoff_.TryDequeueCommand(command))
        {
            break;
        }
        if (!ApplyCommand(command))
        {
            deferred_command_.emplace(std::move(command));
            break;
        }
    }
}

void AudioRuntime::RenderClip() noexcept
{
    if (playback_state_.load(std::memory_order_relaxed) != PlaybackState::Playing || active_clip_ == nullptr)
    {
        return;
    }

    std::size_t output_offset = 0;
    while (output_offset < input_.size())
    {
        const std::size_t available = active_clip_->samples.size() - clip_cursor_;
        const std::size_t count = std::min(input_.size() - output_offset, available);
        for (std::size_t sample = 0; sample < count; ++sample)
        {
            input_[output_offset + sample] += active_clip_->samples[clip_cursor_ + sample];
        }
        output_offset += count;
        clip_cursor_ += count;

        if (clip_cursor_ != active_clip_->samples.size())
        {
            continue;
        }
        if (clip_loop_)
        {
            clip_cursor_ = 0;
            continue;
        }

        clip_cursor_ = 0;
        playback_state_.store(PlaybackState::Stopped, std::memory_order_relaxed);
        PublishEvent({.kind = AudioEventKind::ClipFinished});
        break;
    }
}

AudioRuntime::ProcessingTimes AudioRuntime::ProcessEngines() noexcept
{
    const sfFDN::AudioBuffer input_buffer(kSystemBlockSize, 1, input_);
    sfFDN::AudioBuffer fdn_output_buffer(kSystemBlockSize, 1, fdn_output_);
    sfFDN::AudioBuffer convolution_output_buffer(kSystemBlockSize, 1, convolution_output_);

    const auto convolution_start = std::chrono::steady_clock::now();
    if (active_convolver_ != nullptr)
    {
        active_convolver_->Process(input_buffer, convolution_output_buffer);
    }
    const auto convolution_end = std::chrono::steady_clock::now();

    const auto fdn_start = std::chrono::steady_clock::now();
    active_fdn_->Process(input_buffer, fdn_output_buffer);
    const auto fdn_end = std::chrono::steady_clock::now();

    return {
        .convolution_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(convolution_end - convolution_start).count(),
        .fdn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(fdn_end - fdn_start).count(),
    };
}

void AudioRuntime::DetectInstability() noexcept
{
    if (!std::ranges::any_of(fdn_output_, [](float sample) { return std::abs(sample) > 5.0f; }))
    {
        return;
    }

    PublishEvent({
        .kind = AudioEventKind::FdnInstability,
        .generation = active_fdn_generation_,
    });
    active_fdn_->Clear();
}

void AudioRuntime::PrepareMix(ReverbEngine engine) noexcept
{
    if (engine != ReverbEngine::Fdn)
    {
        return;
    }

    const std::uint32_t delay_samples = direct_delay_ms_.load(std::memory_order_relaxed) * sample_rate_ / 1000;
    if (direct_delay_.GetMaximumDelay() < delay_samples)
    {
        PublishEvent({
            .kind = AudioEventKind::DirectDelayOutOfRange,
            .actual = delay_samples,
            .expected = direct_delay_.GetMaximumDelay(),
        });
    }
    direct_delay_.SetDelay(delay_samples);

    sfFDN::AudioBuffer input_buffer(kSystemBlockSize, 1, input_);
    direct_delay_.Process(input_buffer, input_buffer);
}

void AudioRuntime::MixMeterAndWrite(std::span<float> output_buffer, std::span<float> output, std::size_t num_channels,
                                    float wet, float dry) noexcept
{
    Mix(output, input_, {.output_gain = output_gain_.load(std::memory_order_relaxed), .wet = wet, .dry = dry});
    const MeterReading reading = output_meter_.Measure(output);
    AddMonoToInterleaved(output_buffer, output, num_channels);
    meter_rms_.store(reading.rms, std::memory_order_relaxed);

    float previous_peak = meter_peak_.load(std::memory_order_relaxed);
    while (previous_peak < reading.peak &&
           !meter_peak_.compare_exchange_weak(previous_peak, reading.peak, std::memory_order_relaxed))
    {
    }
    if (reading.clipped)
    {
        meter_clipped_.store(true, std::memory_order_relaxed);
    }
}

void AudioRuntime::UpdateCpuUsage(std::int64_t duration_ns) noexcept
{
    const CpuUsageReading cpu_usage =
        cpu_average_.Push(duration_ns, kSystemBlockSize, static_cast<float>(sample_rate_));
    cpu_usage_.store(cpu_usage.average, std::memory_order_relaxed);

    if (duration_ns <= 0 || static_cast<std::uint64_t>(duration_ns) <= cpu_highwater_mark_ns_)
    {
        return;
    }
    cpu_highwater_mark_ns_ = static_cast<std::size_t>(duration_ns);
    PublishEvent({
        .kind = AudioEventKind::CpuHighWater,
        .actual = static_cast<std::uint64_t>(duration_ns),
        .value = cpu_usage.current,
    });
}

void AudioRuntime::PublishEvent(AudioEvent event) noexcept
{
    if (!handoff_.TryPublishEvent(event))
    {
        event_overflow_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace fdn_sandbox::audio
