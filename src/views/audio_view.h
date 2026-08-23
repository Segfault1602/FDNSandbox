#pragma once

#include "audio/audio_engine.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ImVec2;

namespace fdn_sandbox::views
{

/**
 * @brief Renders audio controls and owns audio presentation state.
 *
 * It operates on AudioEngine only; clip decoding and notification policy remain
 * application responsibilities.
 */
class AudioView final
{
  public:
    void UpdateTelemetry(audio::AudioTelemetry telemetry, float latest_peak, bool clipped, float delta_time);

    std::optional<int> DrawTransportControls(audio::AudioEngine& engine);
    void DrawAudioFileSelector(audio::AudioEngine& engine, const char* label);
    void DrawAudioPlayer(audio::AudioEngine& engine, bool has_reference);
    void DrawAudioDeviceConfiguration(audio::AudioEngine& engine);

    void DrawMeterBar(const ImVec2& size) const;
    void DrawCompactOutputMeter() const;
    float DisplayedCpuUsage() const noexcept;
    bool IsClippingWarningDisplayed() const noexcept;

  private:
    struct OutputMeterDisplayState
    {
        float displayed_peak = 0.0f;
        float peak_hold_timer = 0.0f;
        float clipping_debounce_timer = 0.0f;
        float rms_db = -60.0f;
        float peak_db = -60.0f;
        float meter_fraction = 0.0f;
        bool clipping_warning_displayed = false;
    };

    struct AudioDeviceUiState
    {
        std::vector<std::string> supported_audio_drivers;
        std::vector<std::string> output_devices;
        int selected_audio_driver = 0;
        int selected_output_device = 0;
        bool initialized = false;
        bool play_test_tone = false;
    };

    void DrawAudioMixControls(audio::AudioEngine& engine);
    void DrawConvolutionControls(audio::AudioEngine& engine, bool has_reference);
    void DrawOutputMeter() const;
    void DrawAudioStreamControl(audio::AudioEngine& engine);
    void DrawAudioDriverSelector(audio::AudioEngine& engine);
    void DrawOutputDeviceSelector(audio::AudioEngine& engine);
    void DrawAudioStreamStatus(audio::AudioEngine& engine);

    OutputMeterDisplayState output_meter_display_{};
    AudioDeviceUiState device_ui_{};
    float displayed_cpu_usage_ = 0.0f;
    float cpu_usage_display_timer_ = 0.0f;
    float gain_ = 1.0f;
    float fdn_dry_level_ = 0.5f;
    float fdn_wet_level_ = 0.5f;
    float convolution_wet_level_ = 1.0f;
    std::uint32_t direct_delay_ms_ = 0;
    int selected_audio_file_ = 0;
};

} // namespace fdn_sandbox::views
