#include "app.h"

#include "audio/audio_clip_loader.h"
#include "icons.h"
#include "settings.h"
#include "theme.h"

#include <sffdn/sffdn.h>

#include <boost/dll/runtime_symbol_info.hpp>
#include <imgui.h>
#include <quill/LogMacros.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

constexpr std::array<const char*, 4> kAudioFiles = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav",
                                                    "saxophone.wav"};

void FDNToolboxApp::DrawAudioPlayer()
{
    ImGui::Begin("Audio Player");

    DrawAudioMixControls();
    DrawConvolutionControls();
    DrawOutputMeter();

    ImGui::End();
}

void FDNToolboxApp::DrawTransportControls(int selected_audio_file)
{
    if (ImGui::Button(fdn_sandbox::icons::Impulse))
    {
        audio_engine_.TriggerImpulse();
        LOG_INFO(Settings::Instance().GetLogger(), "Playing impulse response...");
    }

    ImGui::SameLine();
    if (audio_engine_.GetPlaybackState() == fdn_sandbox::audio::PlaybackState::Playing)
    {
        if (ImGui::Button(fdn_sandbox::icons::Stop))
        {
            audio_engine_.StopClip();
        }
        return;
    }

    if (ImGui::Button(fdn_sandbox::icons::Play))
    {
        PlaySelectedAudioFile(selected_audio_file);
    }
}

void FDNToolboxApp::PlaySelectedAudioFile(int selected_audio_file)
{
    const auto audio_path = boost::dll::program_location().parent_path() / "audio" /
                            kAudioFiles.at(static_cast<size_t>(selected_audio_file));
    LOG_INFO(Settings::Instance().GetLogger(), "Playing audio file: {}", audio_path.string());
    auto clip =
        fdn_sandbox::audio::LoadAudioClip(std::filesystem::path(audio_path.string()), Settings::Instance().SampleRate());
    if (clip && audio_engine_.TryInstallClip(std::move(*clip)) && audio_engine_.PlayClip(true))
    {
        return;
    }
    const std::string error = clip ? "Failed to submit the decoded clip." : clip.error();
    LOG_ERROR(Settings::Instance().GetLogger(), "Failed to play audio file: {}", error);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "audio-clip-load-failed",
                              "Failed to play audio clip", error, 6.0);
}

void FDNToolboxApp::DrawAudioFileSelector(int& selected_audio_file, const char* label)
{
    if (ImGui::BeginCombo(label, kAudioFiles.at(static_cast<size_t>(selected_audio_file))))
    {
        for (size_t index = 0; index < kAudioFiles.size(); ++index)
        {
            const bool is_selected = std::cmp_equal(selected_audio_file, index);
            if (ImGui::Selectable(kAudioFiles.at(index), is_selected))
            {
                selected_audio_file = static_cast<int>(index);
                audio_engine_.StopClip();
            }
        }
        ImGui::EndCombo();
    }
}

void FDNToolboxApp::DrawAudioMixControls()
{
    ImGui::PushItemWidth(200);
    static float gain = 1.0f;
    static float fdn_dry_level = 0.5f;
    static float fdn_wet_level = 0.5f;
    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 10.0f, "%.2f"))
    {
        audio_engine_.SetOutputGain(gain);
    }

    if (ImGui::SliderFloat("FDN Direct Level", &fdn_dry_level, 0.0f, 1.0f, "%.2f"))
    {
        audio_engine_.SetFdnMix(fdn_wet_level, fdn_dry_level);
    }

    if (ImGui::SliderFloat("FDN Reverb Level", &fdn_wet_level, 0.0f, 1.0f, "%.2f"))
    {
        audio_engine_.SetFdnMix(fdn_wet_level, fdn_dry_level);
    }

    static uint32_t direct_delay_ms = 0;
    constexpr uint32_t kMinDirectDelay = 0;
    constexpr uint32_t kMaxDirectDelay = 100;
    if (ImGui::SliderScalar("Direct Delay (ms)", ImGuiDataType_U32, &direct_delay_ms, &kMinDirectDelay,
                            &kMaxDirectDelay))
    {
        direct_delay_ms = std::clamp(direct_delay_ms, kMinDirectDelay, kMaxDirectDelay);
        audio_engine_.SetDirectDelayMs(direct_delay_ms);
    }

    ImGui::PopItemWidth();
}

void FDNToolboxApp::DrawConvolutionControls()
{
    if (analysis_workspace_.HasReference())
    {
        int selected_reverb_engine =
            audio_engine_.GetReverbEngine() == fdn_sandbox::audio::ReverbEngine::Fdn ? 0 : 1;
        ImGui::RadioButton("FDN Reverb", &selected_reverb_engine, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Convolution Reverb", &selected_reverb_engine, 1);
        audio_engine_.SetReverbEngine(selected_reverb_engine == 0 ? fdn_sandbox::audio::ReverbEngine::Fdn
                                                                  : fdn_sandbox::audio::ReverbEngine::Convolution);

        static float wet_level = 1.0f;
        ImGui::PushItemWidth(200);

        if (ImGui::SliderFloat("Convolution Level", &wet_level, 0.0f, 1.0f, "%.2f"))
        {
            audio_engine_.SetConvolutionWet(wet_level);
        }
        ImGui::PopItemWidth();
    }
}

void FDNToolboxApp::UpdateUiTelemetry()
{
    constexpr float kMinMeterDb = -60.0f;
    constexpr float kPeakHoldSeconds = 1.0f;
    constexpr float kPeakReleaseDbPerSecond = 20.0f;
    const float delta_time = ImGui::GetIO().DeltaTime;
    const fdn_sandbox::audio::AudioTelemetry telemetry = audio_engine_.ReadTelemetry();
    fdn_session_.ObserveActiveGeneration(telemetry.active_fdn_generation);
    const float rms_value = telemetry.rms;
    const float latest_peak = audio_engine_.ConsumePeak();

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

    if (audio_engine_.ConsumeClipped())
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

void FDNToolboxApp::DrawMeterBar(const ImVec2& size)
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

void FDNToolboxApp::DrawOutputMeter()
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
namespace
{
struct AudioDeviceUiState
{
    std::vector<std::string> supported_audio_drivers;
    std::vector<std::string> output_devices;
    int selected_audio_driver = 0;
    int selected_output_device = 0;
};

void DrawAudioDriverSelector(fdn_sandbox::audio::AudioEngine& engine, AudioDeviceUiState& state)
{
    ImGui::Text("Audio Drivers ");
    ImGui::SameLine();
    const std::string current_driver = engine.GetCurrentAudioDriver();
    if (ImGui::BeginCombo("##Audio Drivers", current_driver.c_str()))
    {
        for (int i = 0; i < state.supported_audio_drivers.size(); ++i)
        {
            const bool is_selected = state.selected_audio_driver == i;
            if (ImGui::Selectable(state.supported_audio_drivers[i].c_str(), is_selected))
            {
                state.selected_audio_driver = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching audio driver to {}",
                         state.supported_audio_drivers[i]);
                engine.SetAudioDriver(state.supported_audio_drivers[i]);
                state.output_devices = engine.GetOutputDevices();
                if (state.selected_output_device >= state.output_devices.size())
                {
                    state.selected_output_device = 0;
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

void DrawOutputDeviceSelector(fdn_sandbox::audio::AudioEngine& engine, AudioDeviceUiState& state)
{
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Output Devices");
    ImGui::SameLine();
    const char* preview =
        state.output_devices.empty() ? "" : state.output_devices[state.selected_output_device].c_str();
    if (ImGui::BeginCombo("##Output Devices", preview))
    {
        for (int i = 0; i < state.output_devices.size(); ++i)
        {
            const bool is_selected = state.selected_output_device == i;
            if (ImGui::Selectable(state.output_devices[i].c_str(), is_selected))
            {
                state.selected_output_device = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching output device to {}", state.output_devices[i]);
                engine.SetOutputDevice(state.output_devices[i]);
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void DrawAudioStreamStatus(fdn_sandbox::audio::AudioEngine& engine)
{
    ImGui::Text("Stream Status: ");
    ImGui::SameLine();
    const auto status_color = engine.IsRunning() ? fdn_sandbox::theme::ColorRole::StatusOk
                                                 : fdn_sandbox::theme::ColorRole::StatusError;
    ImGui::TextColored(fdn_sandbox::theme::Color(status_color),
                       engine.IsRunning() ? "Running" : "Stopped");

    const auto audio_stream_info = engine.GetStreamInfo();
    ImGui::Text("Sample Rate: %d", audio_stream_info.sample_rate);
    ImGui::Text("Buffer Size: %d", audio_stream_info.buffer_size);
    ImGui::Text("Num Output Channels: %d", audio_stream_info.num_output_channels);

    static bool play_test_tone = false;
    if (ImGui::Checkbox("Play Test Tone", &play_test_tone))
    {
        engine.PlayTestTone(play_test_tone);
    }
}
} // namespace

void FDNToolboxApp::DrawAudioDeviceGUI()
{
    static AudioDeviceUiState state{
        .supported_audio_drivers = audio_engine_.GetSupportedAudioDrivers(),
        .output_devices = audio_engine_.GetOutputDevices(),
    };

    DrawAudioDriverSelector(audio_engine_, state);
    DrawOutputDeviceSelector(audio_engine_, state);
    DrawAudioStreamControl();
    DrawAudioStreamStatus(audio_engine_);
}

void FDNToolboxApp::DrawAudioStreamControl()
{
    bool stream_running = audio_engine_.IsRunning();
    if (!ImGui::Checkbox("Start/Stop Audio Stream", &stream_running))
    {
        return;
    }

    if (!stream_running)
    {
        StopAudioStream();
        return;
    }

    StartAudioStream();
}
