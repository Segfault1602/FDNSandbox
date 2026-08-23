#include "audio/audio_engine.h"

#include <stdexcept>
#include <utility>

namespace fdn_sandbox::audio
{

AudioEngine::AudioEngine(AudioEngineConfig config)
    : runtime_(config.sample_rate)
    , manager_(audio_manager::create_audio_manager())
{
    if (manager_ == nullptr)
    {
        throw std::runtime_error("Failed to create audio manager");
    }
}

AudioEngine::~AudioEngine()
{
    Shutdown();
}

bool AudioEngine::Start()
{
    if (shutdown_)
    {
        return false;
    }
    if (manager_->is_audio_stream_running())
    {
        return true;
    }
    return manager_->start_audio_stream(
        audio_stream_option::kOutput,
        [this](std::span<float> output_buffer, std::size_t frame_size, std::size_t num_channels) {
            AudioCallback(output_buffer, frame_size, num_channels);
        },
        static_cast<std::uint32_t>(kSystemBlockSize));
}

void AudioEngine::Stop() noexcept
{
    if (manager_ != nullptr)
    {
        manager_->stop_audio_stream();
    }
}

void AudioEngine::Shutdown() noexcept
{
    if (shutdown_)
    {
        return;
    }
    shutdown_ = true;
    runtime_.StopAcceptingCommands();
    Stop();
    runtime_.DrainOffRealtime();
}

bool AudioEngine::IsRunning() const noexcept
{
    return manager_ != nullptr && manager_->is_audio_stream_running();
}

bool AudioEngine::TryInstallFdn(std::uint64_t generation, std::unique_ptr<sfFDN::FDN> value)
{
    return runtime_.SubmitFdn(generation, std::move(value));
}

bool AudioEngine::TryInstallConvolver(std::unique_ptr<sfFDN::PartitionedConvolver> value)
{
    return runtime_.SubmitConvolver(std::move(value));
}

bool AudioEngine::TryInstallClip(std::unique_ptr<AudioClip> value)
{
    return runtime_.SubmitClip(std::move(value));
}

bool AudioEngine::PlayClip(bool loop)
{
    return runtime_.PlayClip(loop);
}

bool AudioEngine::StopClip()
{
    return runtime_.StopClip();
}

bool AudioEngine::TriggerImpulse()
{
    return runtime_.TriggerImpulse();
}

void AudioEngine::FlushPendingCommands()
{
    runtime_.FlushPendingCommands();
}

void AudioEngine::CollectRetiredObjects()
{
    runtime_.CollectRetiredObjects();
}

std::vector<AudioEvent> AudioEngine::PollEvents()
{
    std::vector<AudioEvent> events;
    AudioEvent event{};
    while (runtime_.TryPopEvent(event))
    {
        events.push_back(event);
    }
    return events;
}

void AudioEngine::SetOutputGain(float value) noexcept
{
    runtime_.SetOutputGain(value);
}

void AudioEngine::SetFdnMix(float wet, float dry) noexcept
{
    runtime_.SetFdnMix(wet, dry);
}

void AudioEngine::SetConvolutionWet(float value) noexcept
{
    runtime_.SetConvolutionWet(value);
}

void AudioEngine::SetDirectDelayMs(std::uint32_t value) noexcept
{
    runtime_.SetDirectDelayMs(value);
}

void AudioEngine::SetReverbEngine(ReverbEngine value) noexcept
{
    runtime_.SetReverbEngine(value);
}

ReverbEngine AudioEngine::GetReverbEngine() const noexcept
{
    return runtime_.GetReverbEngine();
}

PlaybackState AudioEngine::GetPlaybackState() const noexcept
{
    return runtime_.GetPlaybackState();
}

AudioTelemetry AudioEngine::ReadTelemetry() const noexcept
{
    return runtime_.ReadTelemetry();
}

float AudioEngine::ConsumePeak() noexcept
{
    return runtime_.ConsumePeak();
}

bool AudioEngine::ConsumeClipped() noexcept
{
    return runtime_.ConsumeClipped();
}

std::vector<std::string> AudioEngine::GetSupportedAudioDrivers() const
{
    return manager_->get_supported_audio_drivers();
}

std::string AudioEngine::GetCurrentAudioDriver() const
{
    return manager_->get_current_audio_driver();
}

void AudioEngine::SetAudioDriver(std::string_view driver)
{
    manager_->set_audio_driver(driver);
}

std::vector<std::string> AudioEngine::GetOutputDevices() const
{
    return manager_->get_output_devices_name();
}

void AudioEngine::SetOutputDevice(std::string_view device)
{
    manager_->set_output_device(device);
}

audio_stream_info AudioEngine::GetStreamInfo() const
{
    return manager_->get_audio_stream_info();
}

void AudioEngine::PlayTestTone(bool play)
{
    manager_->play_test_tone(play);
}

void AudioEngine::AudioCallback(std::span<float> output_buffer, std::size_t frame_size,
                                std::size_t num_channels) noexcept
{
    if (runtime_.PrepareInput(frame_size).empty())
    {
        return;
    }
    runtime_.Process(output_buffer, frame_size, num_channels);
}

} // namespace fdn_sandbox::audio
