#include "views/audio_view.h"

#include "audio/audio_engine.h"
#include "audio/audio_types.h"
#include "icons.h"
#include "settings.h"
#include "theme.h"
#include "views/window_ids.h"

#include <cstddef>
#include <imgui.h>
#include <quill/LogMacros.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>

namespace fdn_sandbox::views
{
namespace
{
constexpr std::array<const char*, 4> kAudioFiles = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav",
                                                    "saxophone.wav"};
}

void AudioView::DrawAudioPlayer(audio::AudioEngine& engine, bool has_reference)
{
    ImGui::Begin(ids::kAudioPlayer);

    DrawAudioMixControls(engine);
    DrawConvolutionControls(engine, has_reference);
    DrawOutputMeter();

    ImGui::End();
}

std::optional<int> AudioView::DrawTransportControls(audio::AudioEngine& engine)
{
    if (ImGui::Button(fdn_sandbox::icons::Impulse))
    {
        if (engine.TriggerImpulse())
        {
            LOG_INFO(Settings::Instance().GetLogger(), "Playing impulse response...");
        }
        else
        {
            LOG_ERROR(Settings::Instance().GetLogger(), "Audio engine rejected the impulse command");
        }
    }

    ImGui::SameLine();
    if (engine.GetPlaybackState() == audio::PlaybackState::Playing)
    {
        if (ImGui::Button(fdn_sandbox::icons::Stop))
        {
            if (!engine.StopClip())
            {
                LOG_WARNING(Settings::Instance().GetLogger(), "Audio engine rejected the stop command");
            }
        }
        return std::nullopt;
    }

    if (ImGui::Button(fdn_sandbox::icons::Play))
    {
        return selected_audio_file_;
    }
    return std::nullopt;
}

void AudioView::DrawAudioFileSelector(audio::AudioEngine& engine, const char* label)
{
    if (ImGui::BeginCombo(label, kAudioFiles.at(static_cast<size_t>(selected_audio_file_))))
    {
        for (size_t index = 0; index < kAudioFiles.size(); ++index)
        {
            const bool is_selected = std::cmp_equal(selected_audio_file_, index);
            if (ImGui::Selectable(kAudioFiles.at(index), is_selected))
            {
                selected_audio_file_ = static_cast<int>(index);
                if (!engine.StopClip())
                {
                    LOG_WARNING(Settings::Instance().GetLogger(), "Audio engine rejected the stop command");
                }
            }
        }
        ImGui::EndCombo();
    }
}

void AudioView::DrawAudioMixControls(audio::AudioEngine& engine)
{
    ImGui::PushItemWidth(200);
    if (ImGui::SliderFloat("Gain", &gain_, 0.0f, 10.0f, "%.2f"))
    {
        engine.SetOutputGain(gain_);
    }

    if (ImGui::SliderFloat("FDN Direct Level", &fdn_dry_level_, 0.0f, 1.0f, "%.2f"))
    {
        engine.SetFdnMix(fdn_wet_level_, fdn_dry_level_);
    }

    if (ImGui::SliderFloat("FDN Reverb Level", &fdn_wet_level_, 0.0f, 1.0f, "%.2f"))
    {
        engine.SetFdnMix(fdn_wet_level_, fdn_dry_level_);
    }

    constexpr uint32_t kMinDirectDelay = 0;
    constexpr uint32_t kMaxDirectDelay = 100;
    if (ImGui::SliderScalar("Direct Delay (ms)", ImGuiDataType_U32, &direct_delay_ms_, &kMinDirectDelay,
                            &kMaxDirectDelay))
    {
        direct_delay_ms_ = std::clamp(direct_delay_ms_, kMinDirectDelay, kMaxDirectDelay);
        engine.SetDirectDelayMs(direct_delay_ms_);
    }

    ImGui::PopItemWidth();
}

void AudioView::DrawConvolutionControls(audio::AudioEngine& engine, bool has_reference)
{
    if (has_reference)
    {
        int selected_reverb_engine = engine.GetReverbEngine() == audio::ReverbEngine::Fdn ? 0 : 1;
        ImGui::RadioButton("FDN Reverb", &selected_reverb_engine, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Convolution Reverb", &selected_reverb_engine, 1);
        engine.SetReverbEngine(selected_reverb_engine == 0 ? audio::ReverbEngine::Fdn
                                                           : audio::ReverbEngine::Convolution);

        ImGui::PushItemWidth(200);

        if (ImGui::SliderFloat("Convolution Level", &convolution_wet_level_, 0.0f, 1.0f, "%.2f"))
        {
            engine.SetConvolutionWet(convolution_wet_level_);
        }
        ImGui::PopItemWidth();
    }
}

void AudioView::UpdateTelemetry(audio::AudioTelemetry telemetry, float latest_peak, bool clipped, float delta_time)
{
    constexpr float kMinMeterDb = -60.0f;
    constexpr float kPeakHoldSeconds = 1.0f;
    constexpr float kPeakReleaseDbPerSecond = 20.0f;
    const float rms_value = telemetry.rms;

    if (latest_peak >= output_meter_display_.displayed_peak)
    {
        output_meter_display_.displayed_peak = latest_peak;
        output_meter_display_.peak_hold_timer = kPeakHoldSeconds;
    }
    else if (output_meter_display_.peak_hold_timer > 0.0f)
    {
        output_meter_display_.peak_hold_timer = std::max(0.0f, output_meter_display_.peak_hold_timer - delta_time);
    }
    else
    {
        const float release_gain = std::pow(10.0f, -kPeakReleaseDbPerSecond * delta_time / 20.0f);
        output_meter_display_.displayed_peak =
            std::max(latest_peak, output_meter_display_.displayed_peak * release_gain);
    }

    if (clipped)
    {
        output_meter_display_.clipping_warning_displayed = true;
        output_meter_display_.clipping_debounce_timer = 0.0f;
    }
    else
    {
        output_meter_display_.clipping_debounce_timer += delta_time;
    }

    if (output_meter_display_.clipping_warning_displayed && output_meter_display_.clipping_debounce_timer >= 1.0f)
    {
        output_meter_display_.clipping_warning_displayed = false;
    }

    output_meter_display_.rms_db = rms_value > 0.0f ? 20.0f * std::log10(rms_value) : kMinMeterDb;
    output_meter_display_.peak_db = output_meter_display_.displayed_peak > 0.0f
                                        ? 20.0f * std::log10(output_meter_display_.displayed_peak)
                                        : kMinMeterDb;
    output_meter_display_.meter_fraction =
        std::clamp((output_meter_display_.rms_db - kMinMeterDb) / -kMinMeterDb, 0.0f, 1.0f);

    constexpr float kCpuUsageDisplayInterval = 0.25f;
    cpu_usage_display_timer_ += delta_time;
    if (cpu_usage_display_timer_ >= kCpuUsageDisplayInterval)
    {
        displayed_cpu_usage_ = telemetry.cpu_usage;
        cpu_usage_display_timer_ = 0.0f;
    }
}

void AudioView::DrawMeterBar(const ImVec2& size) const
{
    constexpr float kMinMeterDb = -60.0f;
    if (output_meter_display_.clipping_warning_displayed)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError));
    }
    else if (output_meter_display_.rms_db < -12.0f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterSafe));
    }
    else if (output_meter_display_.rms_db < -3.0f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterWarning));
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterHot));
    }

    const std::string meter_overlay = output_meter_display_.rms_db <= kMinMeterDb
                                          ? std::format("< {:.0f} dBFS", kMinMeterDb)
                                          : std::format("{:.1f} dBFS", output_meter_display_.rms_db);
    ImGui::ProgressBar(output_meter_display_.meter_fraction, size, meter_overlay.c_str());
    ImGui::PopStyleColor();

    const float peak_fraction = std::clamp((output_meter_display_.peak_db - kMinMeterDb) / -kMinMeterDb, 0.0f, 1.0f);
    const ImVec2 bar_min = ImGui::GetItemRectMin();
    const ImVec2 bar_max = ImGui::GetItemRectMax();
    const float bar_width = bar_max.x - bar_min.x;
    const float peak_x = bar_min.x + (bar_width * peak_fraction);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(peak_x, bar_min.y), ImVec2(peak_x, bar_max.y),
                                        IM_COL32(235, 238, 242, 210), 1.5f);
}

void AudioView::DrawOutputMeter() const
{
    ImGui::Text("RMS level:");
    ImGui::SameLine();
    DrawMeterBar(ImVec2(200.0f, 0.0f));
    ImGui::Text("Sample peak hold: %.1f dBFS", output_meter_display_.peak_db);
    if (output_meter_display_.clipping_warning_displayed)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "CLIP");
    }
    ImGui::Text("CPU Usage: %.1f%%", displayed_cpu_usage_ * 100.0f);
}
void AudioView::DrawAudioDriverSelector(audio::AudioEngine& engine)
{
    ImGui::Text("Audio Drivers ");
    ImGui::SameLine();
    const std::string current_driver = engine.GetCurrentAudioDriver();
    if (ImGui::BeginCombo("##Audio Drivers", current_driver.c_str()))
    {
        for (int i = 0; i < device_ui_.supported_audio_drivers.size(); ++i)
        {
            const bool is_selected = device_ui_.selected_audio_driver == i;
            if (ImGui::Selectable(device_ui_.supported_audio_drivers[i].c_str(), is_selected))
            {
                device_ui_.selected_audio_driver = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching audio driver to {}",
                         device_ui_.supported_audio_drivers[i]);
                engine.SetAudioDriver(device_ui_.supported_audio_drivers[i]);
                device_ui_.output_devices = engine.GetOutputDevices();
                if (device_ui_.selected_output_device >= device_ui_.output_devices.size())
                {
                    device_ui_.selected_output_device = 0;
                }
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void AudioView::DrawOutputDeviceSelector(audio::AudioEngine& engine)
{
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Output Devices");
    ImGui::SameLine();
    const char* preview =
        device_ui_.output_devices.empty() ? "" : device_ui_.output_devices[device_ui_.selected_output_device].c_str();
    if (ImGui::BeginCombo("##Output Devices", preview))
    {
        for (int i = 0; i < device_ui_.output_devices.size(); ++i)
        {
            const bool is_selected = device_ui_.selected_output_device == i;
            if (ImGui::Selectable(device_ui_.output_devices[i].c_str(), is_selected))
            {
                device_ui_.selected_output_device = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching output device to {}",
                         device_ui_.output_devices[i]);
                engine.SetOutputDevice(device_ui_.output_devices[i]);
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void AudioView::DrawAudioStreamStatus(audio::AudioEngine& engine)
{
    ImGui::Text("Stream Status: ");
    ImGui::SameLine();
    const auto status_color =
        engine.IsRunning() ? fdn_sandbox::theme::ColorRole::StatusOk : fdn_sandbox::theme::ColorRole::StatusError;
    ImGui::TextColored(fdn_sandbox::theme::Color(status_color), engine.IsRunning() ? "Running" : "Stopped");

    const auto audio_stream_info = engine.GetStreamInfo();
    ImGui::Text("Sample Rate: %d", audio_stream_info.sample_rate);
    ImGui::Text("Buffer Size: %d", audio_stream_info.buffer_size);
    ImGui::Text("Num Output Channels: %d", audio_stream_info.num_output_channels);

    if (ImGui::Checkbox("Play Test Tone", &device_ui_.play_test_tone))
    {
        engine.PlayTestTone(device_ui_.play_test_tone);
    }
}

void AudioView::DrawAudioDeviceConfiguration(audio::AudioEngine& engine)
{
    if (!device_ui_.initialized)
    {
        device_ui_.supported_audio_drivers = engine.GetSupportedAudioDrivers();
        device_ui_.output_devices = engine.GetOutputDevices();
        device_ui_.initialized = true;
    }

    DrawAudioDriverSelector(engine);
    DrawOutputDeviceSelector(engine);
    DrawAudioStreamControl(engine);
    DrawAudioStreamStatus(engine);
}

void AudioView::DrawAudioStreamControl(audio::AudioEngine& engine)
{
    bool stream_running = engine.IsRunning();
    if (!ImGui::Checkbox("Start/Stop Audio Stream", &stream_running))
    {
        return;
    }

    if (!stream_running)
    {
        engine.Stop();
        return;
    }

    if (!engine.Start())
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to start audio stream");
        return;
    }
    LOG_INFO(Settings::Instance().GetLogger(), "Audio stream started");
}

void AudioView::DrawCompactOutputMeter() const
{
    ImGui::Text("RMS");
    ImGui::SameLine();
    DrawMeterBar(ImVec2(150.0f, 0.0f));
    if (output_meter_display_.clipping_warning_displayed)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "CLIP");
    }
}

float AudioView::DisplayedCpuUsage() const noexcept
{
    return displayed_cpu_usage_;
}

bool AudioView::IsClippingWarningDisplayed() const noexcept
{
    return output_meter_display_.clipping_warning_displayed;
}

} // namespace fdn_sandbox::views
