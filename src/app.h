#pragma once

#include <atomic>

#include <audio_utils/audio_file_manager.h>
#include <audio_utils/audio_manager.h>

#include "analysis/fdn_analyzer.h"

#include <sffdn/sffdn.h>

enum class DelayFilterType
{
    Proportional = 0,
    OnePole = 1,
    TwoFilter = 2,
};

struct FDNConfig
{
    uint32_t N;
    std::vector<float> input_gains;
    std::vector<float> output_gains;
    std::vector<uint32_t> delays;
    std::vector<float> feedback_matrix;

    // Configuration for cascaded feedback matrix
    bool is_cascaded = false;
    sfFDN::CascadedFeedbackMatrixInfo cascaded_feedback_matrix_info;
    int num_stages = 1;        // Number of stages for cascaded feedback matrix
    float sparsity = 1.0f;     // Sparsity level for cascaded feedback matrix
    float cascade_gain = 1.0f; // Gain per sample for cascaded feedback matrix

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
    void DrawEchoDensity();
    void DrawT60s();

    bool DrawToneCorrectionFilterDesigner(FDNConfig& fdn_config);

    void UpdateFDN();

    void AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels);

    // Member variables
    std::unique_ptr<audio_manager> audio_manager_;
    std::unique_ptr<audio_file_manager> audio_file_manager_;

    bool show_tc_filter_designer_;

    std::unique_ptr<sfFDN::FDN> gui_fdn_;
    std::unique_ptr<sfFDN::FDN> audio_fdn_;
    std::unique_ptr<sfFDN::FDN> other_fdn_;

    FDNConfig fdn_config_;

    enum class AudioState
    {
        Idle,
        ImpulseRequested,

    } audio_state_ = AudioState::Idle;

    std::atomic<float> audio_gain_ = 1.0f;
    std::atomic<float> dry_wet_mix_ = 0.5f;
    std::atomic<float> fdn_cpu_usage_ = 0.0f;

    fdn_analysis::FDNAnalyzer fdn_analyzer_;
};