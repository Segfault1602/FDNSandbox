#pragma once

#include <atomic>
#include <bitset>

#include <sndfile.h>

#include <audio_file_manager.h>
#include <audio_manager.h>
#include <fft_utils.h>

#include <fdn.h>

enum class DelayFilterType
{
    Proportional = 0,
    OnePole = 1,
    TwoFilter = 2,
};

struct FDNConfig
{
    float ir_duration;
    uint32_t sample_rate;
    uint32_t N;
    std::vector<float> input_gains;
    std::vector<float> output_gains;
    std::vector<uint32_t> delays;
    std::vector<float> feedback_matrix;

    DelayFilterType delay_filter_type = DelayFilterType::Proportional;
    float feedback_gain;         // Only used for proportional feedback gains
    float t60_dc;                // Only used for one-pole filters
    float t60_ny;                // Only used for one-pole filters
    std::vector<float> t60s;     // Only used for two-filter design
    std::vector<float> tc_gains; // Tone correction gains, used for GEQ
};

class FDNToolboxApp
{
  public:
    FDNToolboxApp();
    ~FDNToolboxApp();

    FDNToolboxApp(const FDNToolboxApp&) = delete;
    FDNToolboxApp& operator=(const FDNToolboxApp&) = delete;

    void loop();

  private:
    // Functions
    void DrawAudioDeviceGUI();
    bool DrawFDNConfigurator(FDNConfig& fdn_config);
    void DrawImpulseResponse();
    void DrawAudioPlayer();
    void DrawVisualization();
    void DrawSpectrogram();
    void DrawSpectrum();
    void DrawAutocorrelation();
    void DrawFilterResponse();
    void DrawEnergyDecayCurve();
    void DrawCepstrum();

    bool DrawFilterDesigner(FDNConfig& fdn_config);
    bool DrawToneCorrectionFilterDesigner(FDNConfig& fdn_config);

    void UpdateFDN();

    void AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels);

    // Member variables
    std::unique_ptr<audio_manager> audio_manager_;
    std::unique_ptr<audio_file_manager> audio_file_manager_;

    std::vector<float> impulse_response_;
    std::vector<float> spectrogram_data_;
    spectrogram_info spectrogram_info_;
    bool show_delay_filter_designer_;
    bool show_tc_filter_designer_;

    std::unique_ptr<sfFDN::FDN> audio_fdn_;
    std::unique_ptr<sfFDN::FDN> other_fdn_;

    FDNConfig fdn_config_;

    enum class AudioState
    {
        Idle,
        ImpulseRequested,
        PlayingDrums,

    } audio_state_ = AudioState::Idle;

    std::atomic<float> audio_gain_ = 1.0f;
    std::atomic<float> dry_wet_mix_ = 0.5f;
    std::atomic<float> fdn_cpu_usage_ = 0.0f;

    enum WindowType
    {
        ImpulseResponse = 0,
        Spectrogram = 1,
        Spectrum = 2,
        Autocorrelation = 3,
        FilterResponse = 4,
        EnergyDecayCurve = 5,
        Cepstrum = 6,
        WindowTypeCount
    };
    std::bitset<WindowType::WindowTypeCount> impulse_response_changed_ = 0;
};