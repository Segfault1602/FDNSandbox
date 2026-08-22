#include "app.h"

#include "icons.h"
#include "settings.h"
#include "theme.h"

#include <sffdn/sffdn.h>

#include <boost/dll/runtime_symbol_info.hpp>
#include <imgui.h>
#include <quill/LogMacros.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing

void Crossfade(std::span<const float> fade_in, std::span<const float> fade_out, std::span<float> output)
{
    assert(fade_in.size() == fade_out.size());
    assert(fade_in.size() == output.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(output.size() - 1);
        const float fadeout_gain =
            (t * t) * (3.0f - (2.0f * t)); // from https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
        const float fadein_gain = 1.0f - fadeout_gain;
        output[i] = (fade_in[i] * fadein_gain) + (fade_out[i] * fadeout_gain);
    }
}
} // namespace

void FDNToolboxApp::AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels)
{
    if (frame_size != kSystemBlockSize)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Frame size mismatch: expected {}, got {}", kSystemBlockSize,
                  frame_size);
        return;
    }

    ApplyPendingFDNUpdates();
    AdoptPendingConvolutionReverb(frame_size);
    if (audio_fdn_ == nullptr)
    {
        std::ranges::fill(output_buffer, 0.0f);
        return;
    }

    audio_fdn_->SetDirectGain(0.f);

    std::array<float, kSystemBlockSize> input_data{};
    std::array<float, kSystemBlockSize> fdn_output_data{};
    std::array<float, kSystemBlockSize> convolution_output_data{};

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data.front() = 1.0f;
    }

    audio_file_manager_->process_block(input_data, kSystemBlockSize, 1);
    const AudioProcessingTimes processing_times =
        ProcessAudioEngines(input_data, fdn_output_data, convolution_output_data);
    ClearUnstableFDN(fdn_output_data);

    const int reverb_type = reverb_engine_.load();
    if (last_reverb_type_ == kFDN_REVERB && reverb_type == kCONV_REVERB)
    {
        Crossfade(convolution_output_data, fdn_output_data, convolution_output_data);
    }
    else if (last_reverb_type_ == kCONV_REVERB && reverb_type == kFDN_REVERB)
    {
        Crossfade(fdn_output_data, convolution_output_data, fdn_output_data);
    }
    last_reverb_type_ = reverb_type;

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data.front() -= 1.0f;
        audio_state_ = AudioState::Idle;
    }

    const auto [wet_mix, dry_mix] = PrepareMix(reverb_type, input_data);
    const std::span<float> output_data =
        reverb_type == kFDN_REVERB ? std::span<float>(fdn_output_data) : std::span<float>(convolution_output_data);
    MixAndMeter(output_data, input_data, audio_gain_.load(), wet_mix, dry_mix);
    WriteOutput(output_buffer, output_data, num_channels);

    const int64_t duration = reverb_type == kFDN_REVERB ? processing_times.fdn_ns : processing_times.convolution_ns;
    UpdateCpuUsage(duration, frame_size);
}

void FDNToolboxApp::ApplyPendingFDNUpdates()
{
    constexpr size_t kFdnQueueCapacity = 16;
    const size_t queued_updates = std::min(fdn_update_queue_.size_approx(), kFdnQueueCapacity);
    for (size_t i = 0; i < queued_updates; ++i)
    {
        PendingFDNUpdate next_update;
        if (!fdn_update_queue_.try_dequeue(next_update))
        {
            break;
        }
        fdn_cleanup_queue_.emplace(std::move(audio_fdn_));
        audio_fdn_ = std::move(next_update.fdn);
        active_audio_fdn_generation_ = next_update.generation;
    }
}

void FDNToolboxApp::AdoptPendingConvolutionReverb(size_t frame_size)
{
    if (convolution_reverb_ == nullptr)
    {
        return;
    }

    active_convolution_reverb_ = std::move(convolution_reverb_);
    if (active_convolution_reverb_->GetBlockSize() == frame_size)
    {
        return;
    }

    LOG_ERROR(Settings::Instance().GetLogger(),
              "RIR PartitionedConvolver block size {} does not match system block size {}",
              active_convolution_reverb_->GetBlockSize(), frame_size);
    active_convolution_reverb_.reset();
}

FDNToolboxApp::AudioProcessingTimes FDNToolboxApp::ProcessAudioEngines(std::span<float> input,
                                                                       std::span<float> fdn_output,
                                                                       std::span<float> convolution_output)
{
    const sfFDN::AudioBuffer input_buffer(kSystemBlockSize, 1, input);
    sfFDN::AudioBuffer fdn_output_buffer(kSystemBlockSize, 1, fdn_output);
    sfFDN::AudioBuffer convolution_output_buffer(kSystemBlockSize, 1, convolution_output);

    const auto convolution_start = std::chrono::steady_clock::now();
    if (active_convolution_reverb_ != nullptr)
    {
        active_convolution_reverb_->Process(input_buffer, convolution_output_buffer);
    }
    const auto convolution_end = std::chrono::steady_clock::now();

    const auto fdn_start = std::chrono::steady_clock::now();
    audio_fdn_->Process(input_buffer, fdn_output_buffer);
    const auto fdn_end = std::chrono::steady_clock::now();

    return {
        .convolution_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(convolution_end - convolution_start).count(),
        .fdn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(fdn_end - fdn_start).count(),
    };
}

void FDNToolboxApp::ClearUnstableFDN(std::span<const float> fdn_output)
{
    if (!std::ranges::any_of(fdn_output, [](float sample) { return std::abs(sample) > 5.0f; }))
    {
        return;
    }

    LOG_ERROR(Settings::Instance().GetLogger(), "FDN output exceeded |5.0|, likely unstable. Clearing FDN.");
    fdn_instability_generation_.store(active_audio_fdn_generation_, std::memory_order_release);
    audio_fdn_->Clear();
}

std::pair<float, float> FDNToolboxApp::PrepareMix(int reverb_type, std::span<float> input)
{
    if (reverb_type != kFDN_REVERB)
    {
        return {conv_wet_level_.load(), 0.0f};
    }

    const uint32_t delay_samples = direct_delay_ms_.load() * Settings::Instance().SampleRate() / 1000;
    if (direct_delay_.GetMaximumDelay() < delay_samples)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Pre-delay samples {} exceed maximum delay {}", delay_samples,
                  direct_delay_.GetMaximumDelay());
    }
    direct_delay_.SetDelay(delay_samples);

    sfFDN::AudioBuffer input_buffer(kSystemBlockSize, 1, input);
    direct_delay_.Process(input_buffer, input_buffer);
    return {fdn_wet_level_.load(), fdn_dry_level_.load()};
}

void FDNToolboxApp::MixAndMeter(std::span<float> output, std::span<const float> input, float gain, float wet_mix,
                                float dry_mix)
{
    float block_peak = 0.0f;
    for (size_t i = 0; i < output.size(); ++i)
    {
        output[i] = gain * ((wet_mix * output[i]) + (dry_mix * input[i]));
        const float squared_sample = output[i] * output[i];
        meter_sum_squares_ -= static_cast<double>(meter_squared_samples_[meter_sample_index_]);
        meter_squared_samples_[meter_sample_index_] = squared_sample;
        meter_sum_squares_ += static_cast<double>(squared_sample);
        meter_sample_index_ = (meter_sample_index_ + 1) % meter_squared_samples_.size();
        block_peak = std::max(block_peak, std::abs(output[i]));
    }

    const double mean_square = std::max(0.0, meter_sum_squares_) / static_cast<double>(meter_squared_samples_.size());
    meter_rms_.store(std::sqrt(static_cast<float>(mean_square)), std::memory_order_relaxed);

    float previous_peak = meter_peak_.load(std::memory_order_relaxed);
    while (previous_peak < block_peak &&
           !meter_peak_.compare_exchange_weak(previous_peak, block_peak, std::memory_order_relaxed))
    {
    }
    if (block_peak >= 1.0f)
    {
        meter_clipped_.store(true, std::memory_order_relaxed);
    }
}

void FDNToolboxApp::WriteOutput(std::span<float> output_buffer, std::span<const float> output, size_t num_channels)
{
    for (size_t frame = 0; frame < output.size(); ++frame)
    {
        const size_t offset = frame * num_channels;
        for (size_t channel = 0; channel < num_channels; ++channel)
        {
            output_buffer[offset + channel] += output[frame];
        }
    }
}

void FDNToolboxApp::UpdateCpuUsage(int64_t duration_ns, size_t frame_size)
{
    const float allowed_time = (1e9f / Settings::Instance().SampleRateAs<float>()) * static_cast<float>(frame_size);
    const float cpu_usage = static_cast<float>(duration_ns) / allowed_time;
    cpu_usage_sum_ -= cpu_usage_samples_[cpu_usage_sample_index_];
    cpu_usage_samples_[cpu_usage_sample_index_] = cpu_usage;
    cpu_usage_sum_ += cpu_usage;
    cpu_usage_sample_index_ = (cpu_usage_sample_index_ + 1) % kCpuUsageWindowSize;
    cpu_usage_sample_count_ = std::min(cpu_usage_sample_count_ + 1, kCpuUsageWindowSize);
    fdn_cpu_usage_.store(cpu_usage_sum_ / static_cast<float>(cpu_usage_sample_count_), std::memory_order_relaxed);

    if (std::cmp_less_equal(duration_ns, cpu_highwater_mark_ns_))
    {
        return;
    }
    cpu_highwater_mark_ns_ = static_cast<size_t>(duration_ns);
    LOG_WARNING(Settings::Instance().GetLogger(), "New CPU highwater mark: {:.2f}%", cpu_usage * 100.f);
}
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
        audio_state_ = AudioState::ImpulseRequested;
        LOG_INFO(Settings::Instance().GetLogger(), "Playing impulse response...");
    }

    ImGui::SameLine();
    if (audio_file_manager_->get_state() == audio_file_manager::AudioPlayerState::kPlaying)
    {
        if (ImGui::Button(fdn_sandbox::icons::Stop))
        {
            audio_file_manager_->stop(true);
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
    constexpr const char* kAudioFiles[] = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav", "saxophone.wav"};
    const auto audio_path = boost::dll::program_location().parent_path() / "audio" / kAudioFiles[selected_audio_file];
    LOG_INFO(Settings::Instance().GetLogger(), "Playing audio file: {}", audio_path.string());
    if (audio_file_manager_->open_audio_file(audio_path.string()))
    {
        audio_file_manager_->play(true);
        return;
    }
    LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file.");
}

void FDNToolboxApp::DrawAudioFileSelector(int& selected_audio_file, const char* label)
{
    constexpr const char* kAudioFiles[] = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav", "saxophone.wav"};
    if (ImGui::BeginCombo(label, kAudioFiles[selected_audio_file]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(kAudioFiles); i++)
        {
            const bool is_selected = selected_audio_file == i;
            if (ImGui::Selectable(kAudioFiles[i], is_selected))
            {
                selected_audio_file = i;
                audio_file_manager_->stop(true);
            }
        }
        ImGui::EndCombo();
    }
}

void FDNToolboxApp::DrawAudioMixControls()
{
    ImGui::PushItemWidth(200);
    static float gain = 1.0f;
    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 10.0f, "%.2f"))
    {
        audio_gain_ = gain;
    }

    static float fdn_dry_level = 0.5f;
    if (ImGui::SliderFloat("FDN Direct Level", &fdn_dry_level, 0.0f, 1.0f, "%.2f"))
    {
        fdn_dry_level_ = fdn_dry_level;
    }

    static float fdn_wet_level = 0.5f;
    if (ImGui::SliderFloat("FDN Reverb Level", &fdn_wet_level, 0.0f, 1.0f, "%.2f"))
    {
        fdn_wet_level_ = fdn_wet_level;
    }

    uint32_t direct_delay_ms = direct_delay_ms_.load();
    constexpr uint32_t kMinDirectDelay = 0;
    constexpr uint32_t kMaxDirectDelay = 100;
    if (ImGui::SliderScalar("Direct Delay (ms)", ImGuiDataType_U32, &direct_delay_ms, &kMinDirectDelay,
                            &kMaxDirectDelay))
    {
        direct_delay_ms = std::clamp(direct_delay_ms, kMinDirectDelay, kMaxDirectDelay);
        direct_delay_ms_ = direct_delay_ms;
    }

    ImGui::PopItemWidth();
}

void FDNToolboxApp::DrawConvolutionControls()
{
    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        int selected_reverb_engine = reverb_engine_.load();
        ImGui::RadioButton("FDN Reverb", &selected_reverb_engine, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Convolution Reverb", &selected_reverb_engine, 1);
        reverb_engine_.store(selected_reverb_engine);

        static float wet_level = 1.0f;
        ImGui::PushItemWidth(200);

        if (ImGui::SliderFloat("Convolution Level", &wet_level, 0.0f, 1.0f, "%.2f"))
        {
            conv_wet_level_ = wet_level;
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
    const float rms_value = meter_rms_.load(std::memory_order_relaxed);
    const float latest_peak = meter_peak_.exchange(0.0f, std::memory_order_relaxed);

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

    if (meter_clipped_.exchange(false, std::memory_order_relaxed))
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
        displayed_cpu_usage_ = fdn_cpu_usage_.load(std::memory_order_relaxed);
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

void DrawAudioDriverSelector(audio_manager& manager, AudioDeviceUiState& state)
{
    ImGui::Text("Audio Drivers ");
    ImGui::SameLine();
    if (ImGui::BeginCombo("##Audio Drivers", manager.get_current_audio_driver().c_str()))
    {
        for (int i = 0; i < state.supported_audio_drivers.size(); ++i)
        {
            const bool is_selected = state.selected_audio_driver == i;
            if (ImGui::Selectable(state.supported_audio_drivers[i].c_str(), is_selected))
            {
                state.selected_audio_driver = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching audio driver to {}",
                         state.supported_audio_drivers[i]);
                manager.set_audio_driver(state.supported_audio_drivers[i]);
                state.output_devices = manager.get_output_devices_name();
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

void DrawOutputDeviceSelector(audio_manager& manager, AudioDeviceUiState& state)
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
                manager.set_output_device(state.output_devices[i]);
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void DrawAudioStreamStatus(audio_manager& manager)
{
    ImGui::Text("Stream Status: ");
    ImGui::SameLine();
    const auto status_color = manager.is_audio_stream_running() ? fdn_sandbox::theme::ColorRole::StatusOk
                                                                : fdn_sandbox::theme::ColorRole::StatusError;
    ImGui::TextColored(fdn_sandbox::theme::Color(status_color),
                       manager.is_audio_stream_running() ? "Running" : "Stopped");

    const auto audio_stream_info = manager.get_audio_stream_info();
    ImGui::Text("Sample Rate: %d", audio_stream_info.sample_rate);
    ImGui::Text("Buffer Size: %d", audio_stream_info.buffer_size);
    ImGui::Text("Num Output Channels: %d", audio_stream_info.num_output_channels);

    static bool play_test_tone = false;
    if (ImGui::Checkbox("Play Test Tone", &play_test_tone))
    {
        manager.play_test_tone(play_test_tone);
    }
}
} // namespace

void FDNToolboxApp::DrawAudioDeviceGUI()
{
    static AudioDeviceUiState state{
        .supported_audio_drivers = audio_manager_->get_supported_audio_drivers(),
        .output_devices = audio_manager_->get_output_devices_name(),
    };

    DrawAudioDriverSelector(*audio_manager_, state);
    DrawOutputDeviceSelector(*audio_manager_, state);
    DrawAudioStreamControl();
    DrawAudioStreamStatus(*audio_manager_);
}

void FDNToolboxApp::DrawAudioStreamControl()
{
    bool stream_running = audio_manager_->is_audio_stream_running();
    if (!ImGui::Checkbox("Start/Stop Audio Stream", &stream_running))
    {
        return;
    }

    if (!stream_running)
    {
        audio_manager_->stop_audio_stream();
        return;
    }

    if (!audio_manager_->start_audio_stream(
            audio_stream_option::kOutput,
            [this](std::span<float> output_buffer, size_t frame_size, size_t num_channels) {
                AudioCallback(output_buffer, frame_size, num_channels);
            },
            kSystemBlockSize))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to start audio stream");
    }
}
